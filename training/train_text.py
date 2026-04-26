# -*- coding: utf-8 -*-
"""
HybridBERT — Türkçe AI Metin Tespiti Eğitim Scripti
======================================================
Bu script, Türkçe metinlerde yapay zeka tarafından üretilmiş içerikleri
tespit etmek için BERT tabanlı hibrit bir model eğitir.

Mimari:
  - Backbone: dbmdz/bert-base-turkish-cased (Türkçe BERT)
  - Stil katmanı: 12 elle çıkarılan stilometrik özellik → MLP
  - Sınıflandırıcı: BERT [CLS] + Stil MLP → Softmax

12 Stil özelliği:
  1. Ortalama cümle uzunluğu
  2. Cümle uzunluğu standart sapması
  3. Type-Token Ratio (TTR) — kelime çeşitliliği
  4. Virgül oranı
  5. Noktalama işareti oranı
  6. Paragraf sayısı
  7. Ortalama paragraf uzunluğu
  8. Ortalama kelime uzunluğu
  9. Bigram tekrar oranı
  10. AI bağlaç sıklığı (ayrıca, bunun yanı sıra, vb.)
  11. Toplam kelime sayısı
  12. Toplam cümle sayısı
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import re
from transformers import (
    BertTokenizer,
    BertModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. AYARLAR VE CİHAZ
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "dbmdz/bert-base-turkish-cased"  # Türkçe BERT modeli
output_folder = "./ai_buster_final_yeni3"     # Eğitilmiş model çıktı klasörü

# ============================================================
# 2. VERİ YÜKLEME VE TEMİZLİK
# ============================================================
# Excel dosyasından metin ve etiket sütunlarını yükle
df = pd.read_excel('C:\\Users\\aunlu\\OneDrive\\Desktop\\hollyshuffledsheet.xlsx')
df = df.rename(columns={'metin': 'text', 'etiket': 'label'})  # Sütun adlarını standartlaştır
df = df[['text', 'label']].dropna().reset_index(drop=True)    # Eksik değerleri at
df['label'] = df['label'].astype(int)                          # Etiketi int'e çevir (0=insan, 1=AI)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)  # Yinelenen metinleri kaldır


# ============================================================
# 3. STİL ÖZELLİKLERİ FONKSİYONU (12 ÖZELLİK)
# ============================================================
def extract_features(text):
    """
    Bir metnin stilometrik özelliklerini çıkarır.

    Bu özellikler, AI üretimi metinlerin insan metinlerinden farklı
    yazım kalıplarına sahip olduğu gözleminden yola çıkar:
    - AI metinler genellikle daha uniform cümle uzunluklarına sahiptir
    - Bağlaçları (ayrıca, dolayısıyla vb.) aşırı kullanır
    - Bigram tekrarları daha yüksektir (formüle ifadeler)

    Returns:
        list: 12 float değerden oluşan özellik vektörü
    """
    # Cümleleri nokta, ünlem ve soru işaretlerine göre böl
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    words     = text.split()
    num_words = max(len(words), 1)
    num_sents = max(len(sentences), 1)

    # Cümle uzunluğu istatistikleri
    sent_lens    = [len(s.split()) for s in sentences] or [0]
    avg_sent_len = float(np.mean(sent_lens))   # Ortalama cümle uzunluğu
    std_sent_len = float(np.std(sent_lens))    # Std sapma — yüksek = değişken; AI düşük gösterir

    # Type-Token Ratio (TTR): Benzersiz kelime / toplam kelime
    # Yüksek TTR → zengin kelime dağarcığı, düşük TTR → tekrarcı dil
    ttr = len(set(w.lower() for w in words)) / num_words

    # Noktalama kullanım oranları
    comma_rate = text.count(',') / num_words
    punct_rate = sum(text.count(p) for p in '.,;:!?') / num_words

    # Paragraf yapısı
    paragraphs   = [p.strip() for p in text.split('\n') if len(p.strip()) > 10]
    num_paras    = len(paragraphs) or 1
    avg_para_len = float(np.mean([len(p.split()) for p in paragraphs])) if paragraphs else 0.0

    # Ortalama kelime uzunluğu
    avg_word_len = float(np.mean([len(w) for w in words]))

    # Bigram tekrar oranı: Tekrar eden ikili kelime çiftlerinin oranı
    # AI metinler formüle ifadeler nedeniyle daha yüksek bigram tekrarı gösterir
    bigrams       = list(zip(words[:-1], words[1:]))
    bigram_repeat = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    # AI bağlaç sıklığı: Akademik/yapay metinlerde sık görülen Türkçe bağlaçlar
    ai_connectors = [
        'ayrıca', 'bunun yanı sıra', 'öte yandan', 'sonuç olarak',
        'bu bağlamda', 'bu doğrultuda', 'nitekim', 'dolayısıyla'
    ]
    connector_rate = sum(text.lower().count(c) for c in ai_connectors) / num_sents

    return [
        avg_sent_len,    # 1. Ortalama cümle uzunluğu
        std_sent_len,    # 2. Cümle uzunluğu std sapması
        ttr,             # 3. Type-Token Ratio
        comma_rate,      # 4. Virgül oranı
        punct_rate,      # 5. Genel noktalama oranı
        num_paras,       # 6. Paragraf sayısı
        avg_para_len,    # 7. Ortalama paragraf uzunluğu
        avg_word_len,    # 8. Ortalama kelime uzunluğu
        bigram_repeat,   # 9. Bigram tekrar oranı
        connector_rate,  # 10. AI bağlaç sıklığı
        num_words,       # 11. Toplam kelime sayısı
        num_sents,       # 12. Toplam cümle sayısı
    ]


# ============================================================
# Stil özelliklerini çıkar ve standardize et
# ============================================================
print("Özellikler çıkarılıyor ve normalize ediliyor...")
features = np.array([extract_features(t) for t in df['text'].values], dtype=np.float32)
features = np.nan_to_num(features)  # NaN değerleri 0 ile değiştir (güvenlik önlemi)

# StandardScaler: Her özelliği ortalama=0, std=1 olacak şekilde ölçeklendirir
# Bu önemlidir — farklı birimler (kelime sayısı vs oran) direkt karşılaştırılamaz
scaler = StandardScaler()
features = scaler.fit_transform(features).astype(np.float32)

# ============================================================
# 4. EĞİTİM/DOĞRULAMA/TEST AYIRIMI
# ============================================================
# %80 eğitim, %10 doğrulama, %10 test
# stratify=label ile sınıf dağılımı her set'te korunur
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# İlgili satırlara ait özellik tensörleri
train_features = torch.tensor(features[train_df.index.tolist()])
val_features   = torch.tensor(features[val_df.index.tolist()])
test_features  = torch.tensor(features[test_df.index.tolist()])

# ============================================================
# 5. DATASET HAZIRLIĞI (HuggingFace Datasets formatı)
# ============================================================
tokenizer = BertTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    """
    Metinleri BERT tokenizer ile işler.
    padding=False: DataCollatorWithPadding batch bazında dinamik padding yapar
    truncation=True: 256 tokendan uzun metinler kesilir
    """
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=256)


def make_dataset(dataframe, feat_tensor):
    """
    Pandas DataFrame'den HuggingFace Dataset oluşturur ve
    stil özelliklerini (12 boyutlu vektör) ekler.
    """
    ds = Dataset.from_pandas(dataframe[['text', 'label']], preserve_index=False)
    ds = ds.map(tokenize_function, batched=True)          # Tokenizasyon (toplu işlem)
    ds = ds.add_column("style_features", feat_tensor.tolist())  # Stil özelliklerini ekle
    ds.set_format("torch")                                 # PyTorch tensor formatına çevir
    return ds


train_ds = make_dataset(train_df, train_features)
val_ds   = make_dataset(val_df, val_features)
test_ds  = make_dataset(test_df, test_features)


# ============================================================
# 6. MODEL MİMARİSİ
# ============================================================
class HybridBert(nn.Module):
    """
    BERT + Stil Özellikleri Hibrit Sınıflandırıcı.

    İki bilgi kaynağını birleştirir:
    1. BERT [CLS] token: Derin anlamsal temsil (768 boyut)
    2. Stil MLP çıktısı: 12 stil özelliği → 4 boyutlu özet

    Birleşik temsil (768+4=772 boyut) → Sınıflandırıcı → 2 sınıf

    Neden küçük stil MLP?
    - Stil özelliklerinin etkisi kasıtlı olarak kısıtlandı (12→4)
    - Böylece model bunları ezberlemek yerine BERT ile birlikte kullanır
    - Dropout 0.4 ile ek regularization
    """
    def __init__(self, model_name, num_style_features, num_labels=2):
        super().__init__()
        # BERT backbone: Türkçe metni bağlamsal vektörlere dönüştürür
        self.bert   = BertModel.from_pretrained(model_name)
        self.config = self.bert.config  # Trainer için gerekli (hidden_size vb.)

        # Stil özellik MLP: Küçük tutularak ezber önleniyor
        # 12 → 4 (dar boyun — bilgi sıkıştırma)
        self.style_mlp = nn.Sequential(
            nn.Linear(num_style_features, 4),  # 12 → 4
            nn.ReLU(),
            nn.Dropout(0.4),  # Yüksek dropout = stil özelliklerine az güven
        )

        # Ana sınıflandırıcı: BERT (768) + Stil (4) = 772 boyut → 2 sınıf
        # 128 gizli katman ile orta düzey kapasite (daha yüksek → daha fazla ezber)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + 4, 64),  # 772 → 64
            nn.ReLU(),
            nn.Dropout(0.5),   # Yüksek dropout = daha az ezber
            nn.Linear(64, num_labels),  # 64 → 2 (insan / AI)
        )

    def forward(self, input_ids, attention_mask, style_features,
                token_type_ids=None, labels=None, **kwargs):
        """
        Forward pass:
        1. BERT'ten [CLS] token vektörünü al (768 boyut)
        2. Stil özelliklerini MLP'den geçir (4 boyut)
        3. İkisini birleştir ve sınıflandır

        labels verilmişse kayıp hesaplanır (Trainer uyumluluğu için).
        """
        # BERT kodlaması — sadece [CLS] token'ı kullan (ilk token)
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]  # [Batch, 768] — [CLS] vektörü

        # Stil özelliklerini işle
        sf        = style_features.to(cls.device).float()  # GPU'ya taşı
        style_out = self.style_mlp(sf)                     # [Batch, 4]

        # İki temsili birleştir ve sınıflandır
        combined = torch.cat([cls, style_out], dim=1)  # [Batch, 772]
        logits   = self.classifier(combined)            # [Batch, 2]

        # Kayıp hesabı (Trainer için gerekli)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# Model oluştur ve GPU'ya taşı
model = HybridBert(model_name, features.shape[1]).to(device)


# ============================================================
# 7. METRİKLER VE EĞİTİM AYARLARI
# ============================================================
def compute_metrics(p):
    """
    HuggingFace Trainer için metrik hesaplama fonksiyonu.
    Her validation epoch sonunda çağrılır.
    """
    logits, labels = p
    pred = np.argmax(logits, axis=1)  # En yüksek logit → sınıf tahmini
    acc  = accuracy_score(labels, pred)
    _, _, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir="./checkpoints",          # Checkpoint kayıt klasörü
    num_train_epochs=4,                  # 4 epoch: az eğitimde BERT için yeterli
    per_device_train_batch_size=16,      # Batch başına 16 örnek
    learning_rate=2e-5,                  # BERT fine-tuning için standart LR aralığı (1e-5 ~ 5e-5)
    warmup_steps=200,                    # İlk 200 adımda LR lineer artış
    eval_strategy="epoch",              # Her epoch sonunda doğrulama
    save_strategy="epoch",              # Her epoch sonunda kaydet
    load_best_model_at_end=True,        # En iyi checkpoint ile bitir
    label_smoothing_factor=0.1,         # %10 label smoothing — overconfidence önler
    fp16=True,                          # FP16 karışık hassasiyet — hız kazanımı
    report_to="none",                   # WandB/MLflow vb. raporlamayı devre dışı bırak
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    # DataCollator: Farklı uzunluktaki metinleri batch bazında padding ile hizalar
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    # 2 epoch iyileşme yoksa erken durdur (patience=2)
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ============================================================
# 8. EĞİTİMİ BAŞLAT VE MODELİ KAYDET
# ============================================================
if __name__ == '__main__':
    print("Eğitim başlıyor...")
    trainer.train()

    print("\nModel ve ilgili dosyalar kaydediliyor...")
    os.makedirs(output_folder, exist_ok=True)

    # Model ağırlıklarını kaydet (yalnızca state_dict — daha küçük dosya)
    model.to('cpu')  # CPU'ya al (büyük checkpoint'larda GPU belleği serbest bırakılır)
    torch.save(model.state_dict(), f"{output_folder}/state_dict.pth")

    # Tokenizer'ı kaydet (inference sırasında aynı tokenizasyonu sağlar)
    tokenizer.save_pretrained(output_folder)

    # Scaler'ı kaydet (inference sırasında stil özelliklerini aynı şekilde normalize etmek için)
    pickle.dump(scaler, open(f"{output_folder}/scaler.pkl", "wb"))

    print(f"✅ İşlem Tamam! Model şu klasörde: {output_folder}")
    print(f"   - state_dict.pth: Model ağırlıkları")
    print(f"   - tokenizer.*:    Tokenizer dosyaları")
    print(f"   - scaler.pkl:     Stil özelliği normalize edici")
