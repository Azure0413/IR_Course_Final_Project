import os
import torch
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
import re
import torch.nn.functional as F
from nltk.stem import PorterStemmer

# 路徑配置
XML_FOLDER = "./xml_folder"
IMAGE_FOLDER = os.path.join(XML_FOLDER, "image")
RECIPE_FOLDER = os.path.join(XML_FOLDER, "recipe")
ps = PorterStemmer()

# 延遲加載模型與處理器
def get_model_and_processor():
    # 在首次調用時加載模型
    if not hasattr(get_model_and_processor, "model") or not hasattr(get_model_and_processor, "processor"):
        model_path = "./IRW/model/clip_finetuned_cpu"
        get_model_and_processor.model = CLIPModel.from_pretrained(model_path)
        get_model_and_processor.processor = CLIPProcessor.from_pretrained(model_path)
        print("Model loaded!")
    
    return get_model_and_processor.model, get_model_and_processor.processor

# 預處理所有圖片，生成嵌入特徵
def preprocess_images(image_folder, model, processor):
    image_embeddings = []
    image_paths = []
    for image_name in os.listdir(image_folder):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_folder, image_name)
            image = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
            image_embeddings.append(image_features)
            image_paths.append(image_name)
    return image_embeddings, image_paths

# 提取文本特徵
def get_text_features(text_input, model, processor):
    text_inputs = processor(text=[text_input], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    return text_features

# 提取圖片特徵
def get_image_features(image_input, model, processor):
    if isinstance(image_input, str):  # If input is a file path
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):  # If input is a PIL Image object
        image = image_input
    else:
        raise ValueError("Invalid image input type. Expected file path or PIL.Image.Image.")
    image_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
    return image_features

# 特徵融合（文字和圖片）
def fuse_features(text_features, image_features, alpha=0.5):
    return alpha * text_features + (1 - alpha) * image_features

# 匹配前10相似的結果
def match_features_to_images(input_features, image_embeddings, image_paths, top_k=10):
    similarities = [
        torch.nn.functional.cosine_similarity(input_features, image_feature).item()
        for image_feature in image_embeddings
    ]
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [(image_paths[i], similarities[i]) for i in sorted_indices]

# 加載索引文件
def load_index(index_path):
    index_dict = {}
    with open(index_path, "r", encoding='utf-8') as f:
        for line in f:
            image_id, recipe_title = line.strip().split(" ", 1)
            index_dict[image_id] = recipe_title
    return index_dict

index_data = load_index(os.path.join(XML_FOLDER, "index.txt"))

# Text(SBERT)

def get_topk_results(query,model,k):
    ## open recipe embeddings
    with open('./IRW/model/recipe_embeddings.json', "r", encoding="utf-8") as f:recipe_embeddings = json.load(f)
    print("Embeddings Loaded")

    topk = []
    highest_similarity = -1
    best_match_file = None

    query_embedding = model.encode(query)
    for filename, content_embedding in recipe_embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding),torch.tensor(content_embedding),dim=0).item()
        # if similarity > highest_similarity:
        #     highest_similarity = similarity
        #     best_match_file = filename
        #     print(best_match_file,similarity)
        topk.append((filename, similarity))
    topk_sorted = sorted(topk, key=lambda x: x[1], reverse=True)
    name_to_index = {v: k for k, v in index_data.items()}
    topk_with_indices = [(name_to_index[name]+'.jpg', score) for name, score in topk_sorted]

    return topk_with_indices[:k]

def recipe_name_to_index(target):
    key = next((k for k, v in index_data.items() if v == target), None)
    return key
    

def index_view(request):
    query = request.POST.get("q", "").strip()  # 使用 POST 方法取得 query
    method = request.POST.get("method", "1")  # 1: Text, 2: Image, 3: Text & Image
    print(method)

    # 讀取 index.txt 檔案，將其轉為字典
    index_file = os.path.join(XML_FOLDER, "index.txt")
    recipe_names = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                recipe_names[parts[0]] = parts[1]  # key: 檔案名稱, value: 菜名

    results = []
    if query or ("image" in request.FILES):  # 只有當有查詢或上傳圖片時才進行處理
        # 應用延遲加載模型
        model, processor = get_model_and_processor()
        # 預處理圖片嵌入特徵，只在需要的時候加載圖片
        image_embeddings, image_paths = preprocess_images(IMAGE_FOLDER, model, processor)
        sbert_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

        if method == "1" and query:  # Text
            text_features = get_text_features(query, model, processor)
            matched_results = match_features_to_images(text_features, image_embeddings, image_paths)
        elif method == "2" and "image" in request.FILES:  # Image
            uploaded_image = request.FILES["image"]
            image = Image.open(uploaded_image).convert("RGB")
            image_features = get_image_features(image, model, processor)
            matched_results = match_features_to_images(image_features, image_embeddings, image_paths)
        elif method == "3" and query and "image" in request.FILES:  # Text & Image
            text_features = get_text_features(query, model, processor)
            uploaded_image = request.FILES["image"]
            image = Image.open(uploaded_image).convert("RGB")
            image_features = get_image_features(image, model, processor)
            fused_features = fuse_features(text_features, image_features)
            matched_results = match_features_to_images(fused_features, image_embeddings, image_paths)
        elif method == '4' and query: # Text(Sentence Transformer)
            print("Method 4")
            sbert_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            matched_results = get_topk_results(query,sbert_model,10)
            print(matched_results)

        else:
            matched_results = []

        # 組織結果數據
        for image_name, similarity in matched_results:
            print(image_name)
            # 獲取對應的菜名
            recipe_id = image_name.split(".")[0]
            recipe_name = recipe_names.get(recipe_id, "Unknown Recipe")

            # 獲取對應的食譜內容
            recipe_file = os.path.join(RECIPE_FOLDER, f"{recipe_id}.txt")
            if os.path.exists(recipe_file):
                with open(recipe_file, "r", encoding="utf-8") as f:
                    recipe_content = f.read()
            else:
                recipe_content = "Recipe not found."

            results.append({
                "index": recipe_id,  # 傳遞索引值
                "image_url": os.path.join(IMAGE_FOLDER, image_name),
                "recipe_name": recipe_name,  # 新增菜名資訊
                "recipe_content": recipe_content,
                "similarity": similarity,
            })

    return render(request, "index.html", {"results": results, "query": query, "method": method})

# Utility functions for statistics
def count_sentences(text):
    abbreviations = r'\b(?:U\.S\.|e\.g\.|i\.e\.|Dr\.|Mr\.|Ms\.|Prof\.|Ltd\.|Inc\.|Jr\.|Sr\.)\b'
    abbreviation_pattern = re.compile(abbreviations, re.IGNORECASE)
    sentence_endings = re.finditer(r'[.!?]', text)

    if text == "":
        return 0

    sentence_count = 0

    for match in sentence_endings:
        end_pos = match.start()
        if end_pos + 1 < len(text) and text[end_pos + 1] == ' ' and end_pos + 2 < len(text) and (text[end_pos + 2].isupper() or text[end_pos + 2].isdigit()):
            before_punctuation = text[:end_pos].strip().split()[-1] if text[:end_pos].strip() else ''
            if not abbreviation_pattern.search(before_punctuation):
                sentence_count += 1

    sentence_count += 1
    return sentence_count

def count_letters_spaces(text):
    return sum(1 for char in text if char.isalpha())

def count_stemmed_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(set(words))

def count_ascii_non_ascii(text):
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return ascii_count, non_ascii_count

def calculate_statistics(text):
    num_chars = count_letters_spaces(text)
    num_words = count_stemmed_words(text)
    num_sentences = count_sentences(text)
    num_ascii, num_non_ascii = count_ascii_non_ascii(text)

    return {
        'num_chars': num_chars,
        'num_words': num_words,
        'num_sentences': num_sentences,
        'num_ascii': num_ascii,
        'num_non_ascii': num_non_ascii,
    }

# New file_analysis_view
def file_analysis_view(request, recipe_id):
    # 從 index.txt 中獲取菜名
    index_file = os.path.join(XML_FOLDER, "index.txt")
    recipe_name = None
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2 and parts[0] == recipe_id:
                recipe_name = parts[1]
                break

    # 獲取食譜內容
    recipe_file = os.path.join(RECIPE_FOLDER, f"{recipe_id}.txt")
    if os.path.exists(recipe_file):
        with open(recipe_file, "r", encoding="utf-8") as f:
            recipe_content = f.read()
    else:
        recipe_content = "Recipe not found."

    # 構造圖片的完整 URL
    image_file = os.path.join(IMAGE_FOLDER, f"{recipe_id}.jpg")
    if os.path.exists(image_file):
        recipe_image = os.path.join("../../", image_file ) # Update this path based on your static file setup
    else:
        recipe_image = None  # Handle missing images gracefully

    recipe_statstic = calculate_statistics(recipe_content)

    # 返回渲染的頁面
    return render(request, "file_analysis.html", {
        "recipe_name": recipe_name or "Unknown Recipe",
        "recipe_content": recipe_content,
        "statistics": recipe_statstic,
        "recipe_image": recipe_image,  # Add the image to the context
    })


