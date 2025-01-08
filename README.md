# IR_Course_Final_Project

### Reproduction
#### Go here to download the [model file](https://drive.google.com/drive/folders/18O5LDo0gaXhbQolfhH320TDe5BeIiQtv) and put it in ./IRW/model

### 介紹：
#### 本系統以西餐為主題，並包含了約400筆的食譜資料，主要功能包括以食材的文字搜尋菜色的圖片、以食材圖片搜尋相關菜色以及混合食材的文字和圖片找到混合兩者語意的對應菜色圖片，模型部分主要使用了Sentence Transformer, CLIP, BLIP等，系統可提供多種的搜尋方式並會由高到低排序最相關的食譜，每個食譜中也有顯示字元數、字數、句子數、ACSII、non-ACSII的統計數值。

### 功能：
#### 1.	使用CLIP進行查詢，方法包括
#### 	文字輸入搜尋對應圖片
#### 	圖片輸入搜尋對應圖片
#### 	文字+圖片輸入對應圖片
#### 2.	使用Sentence Transformer, Resnet進行食材搜尋
#### 	文字輸入搜尋相關食材圖片
#### 	圖片輸入搜尋相關食材圖片
#### 3.	實作類似LDRE模型，先使用BLIP對圖片產生Caption，並將Caption與prompt用LLM（使用EleutherAI/gpt-neo-1.3B）結合，再透過結合好的新Prompt去計算similarity
#### 4.	食譜的統計數值：字元數、字數、句子數、ACSII、non-ACSII等

### 演算法：
#### 1.	食譜搜尋：CLIP
#### 2.	食材搜尋：Sentence Transformer, Resnet
#### 3.	多模態搜尋：CLIP, LDRE-D
#### 4.	文本處理：Stemming algorithm, Stop-words, Regular Expression
### 心得：
#### 1.	透過CLIP實驗比較發現，有fine tuning的表現較好，其他unfreeze最後幾層的fine tuning效果最好。
#### 2.	LDRE-D實作時有發現，因為最後還是以Text to Image的retrieval方式，因此效果有時不會比單純Image to Image好。
