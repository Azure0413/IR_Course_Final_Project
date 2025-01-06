import os
import torch
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
import re
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from torchvision.models import resnet50
import torchvision.transforms as transforms

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

# SBERT
def get_topk_results(method,query,model,k):
    # encode query

    if method == '4':
        embeddings_path = './IRW/model/recipe_embeddings.json'
        query_embedding = model.encode(query) # text
    elif method == '6':
        embeddings_path = './IRW/model/ingredient_embeddings.json'
        query_embedding = model.encode(query) # text
    elif method == '7':
        embeddings_path = './IRW/model/ingredient_img_embeddings.json'
        query = transform(query).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model(query).squeeze() # image tensor
            query_embedding = query_embedding.flatten()

    with open(embeddings_path, "r", encoding="utf-8") as f:embeddings = json.load(f)
    topk = []
    for filename, content_embedding in embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(query_embedding),torch.tensor(content_embedding),dim=0).item()
        topk.append((filename, similarity))
    topk_sorted = sorted(topk, key=lambda x: x[1], reverse=True)
    

    name_to_index = {v: k for k, v in index_data.items()}  
    topk_with_indices = [(name_to_index[name] + '.jpg' if method == '4' else name + '.jpg', score)
                            for name, score in topk_sorted]
    print(topk_with_indices[:k])
    return topk_with_indices[:k]

def recipe_name_to_index(target):
    key = next((k for k, v in index_data.items() if v == target), None)
    return key

# Resnet
# Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tuned CLIP 模型
clip_processor = CLIPProcessor.from_pretrained("./IRW/model/clip_finetuned_cpu")
clip_model = CLIPModel.from_pretrained("./IRW/model/clip_finetuned_cpu").to("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: 讀取 index.txt 並提取文本描述
with open(os.path.join(XML_FOLDER, "index.txt"), "r", encoding="utf-8") as f:
    text_descriptions = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# 預先生成文本嵌入特徵
text_embeddings = []
for text in text_descriptions:
    text_inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        text_features = clip_model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
    text_embeddings.append(text_features)

# Step 3: 定義圖像轉文字敘述的函數（BLIP）
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Step 4: 定義圖像檢索文本的匹配函數（CLIP）
def match_image_to_text(image_features, text_embeddings, text_descriptions):
    similarities = [
        torch.nn.functional.cosine_similarity(image_features, text_feature).item()
        for text_feature in text_embeddings
    ]
    best_match_idx = similarities.index(max(similarities))
    return text_descriptions[best_match_idx]

# Step 5: 定義圖像轉 CLIP 特徵的函數
def image_to_text(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
    return image_features

# Step 6: 定義 LLM 處理的函數
def refine_query_with_llm(query_text, image_caption):
    # 替換為 Hugging Face 支持的模型名稱
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    # 如果 tokenizer 沒有 pad_token，就將其設為 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 建立要傳遞給 LLM 的提示
    llm_input = (
        # f"I have an ingredient '{query_text}'. "
        # f"And I have the other ingredient '{image_caption}'. "
        # f"I want to eat something with {query_text} and {image_caption}."
        "I willput my image content beginning with 'Image Content:'. The instruction I provide will begin with 'Instruction:'. The edited description you generate should begin with 'Edited Description:'. You just generate one edited description only beginning with 'Edited Description:'. The edited description needs to be as simple as possible. Just one line."
        "An example: Image Content: sliced pizza with cheese. Instruction: apple. Edited Description: I want to eat sliced apple pizza with cheese."
        f"Image Content: {image_caption}"
        f"Instruction: {query_text}"
        f"Edited Description:"
    )

    # 將提示轉為 LLM 的輸入格式
    inputs = tokenizer(llm_input, return_tensors="pt", truncation=True, padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用 LLM 生成回應
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=200,  # 可根據需要調整生成文字的最大長度
            temperature=0.7,  # 控制生成的隨機性
            num_beams=5,  # 使用 beam search 提高生成質量
            no_repeat_ngram_size=2,  # 避免重複的 n-gram
            pad_token_id=tokenizer.eos_token_id  # 使用 eos_token 來填充
        )
    
    refined_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_query   

def index_view(request):
    query = request.POST.get("q", "").strip()  # 使用 POST 方法取得 query
    method = request.POST.get("method", "1")  # 1: Text, 2: Image, 3: Text & Image, 4: SBERT, 5: LLM + BLIP

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
        print(f'method {method}')
        if method == '4' or method == '6':
            model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        elif method == '7':
            model = resnet50(pretrained=True)
            model.eval() 
            model = torch.nn.Sequential(*list(model.children())[:-1])
        else: 
            # 應用延遲加載模型
            model, processor = get_model_and_processor()
            # 預處理圖片嵌入特徵，只在需要的時候加載圖片
            image_embeddings, image_paths = preprocess_images(IMAGE_FOLDER, model, processor)

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
        elif method == '4' and query: # SBERT (Recipe)
            matched_results = get_topk_results(method,query, model, 10)
        elif method == "5" and query and "image" in request.FILES:  # LLM + BLIP + CLIP
            uploaded_image = request.FILES["image"]
            image_caption = generate_caption(uploaded_image)
            print(f"Image Caption: {image_caption}")
            refined_query = refine_query_with_llm(query, image_caption)
            refined_query = f"I want to eat something with {query} and {image_caption}."
            print(f"Refined Query: {refined_query}")
            text_features = get_text_features(refined_query, model, processor)
            matched_results = match_features_to_images(text_features, image_embeddings, image_paths)
        elif method == '6' and query: # SBERT (Ingredients)
            matched_results = get_topk_results(method,query, model, 10)
        elif method == '7' and "image" in request.FILES: # Resnet (Ingredients)
            uploaded_image = request.FILES["image"]
            image = Image.open(uploaded_image).convert("RGB")
            matched_results = get_topk_results(method,image, model, 10)
        else:
            matched_results = []

        for image_name, similarity in matched_results:
            recipe_id = image_name.split(".")[0]
            recipe_name = recipe_names.get(recipe_id, "Unknown Recipe")
            if method == '6' or method == '7': 
                recipe_name = image_name[:-4]
                IMAGE_FOLDER = os.path.join(XML_FOLDER, "Ingredients")
            print(recipe_name)
            print(IMAGE_FOLDER)
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
                "similarity": round(similarity, 3),
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


