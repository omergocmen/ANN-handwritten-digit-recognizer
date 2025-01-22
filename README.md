# El Yazısı Rakam Tanıma Uygulaması

Bu proje, sıfırdan oluşturulmuş bir yapay sinir ağı kullanarak el yazısı rakamları tanıyan basit bir uygulamadır. MNIST veri seti üzerinde eğitilmiş olan model, kullanıcının çizdiği rakamları gerçek zamanlı olarak tanıyabilmektedir.

## Özellikler

- Sıfırdan kodlanmış yapay sinir ağı
- İnteraktif çizim arayüzü
- Gerçek zamanlı tahmin
- Tahmin güven skorları
- Otomatik görüntü önişleme ve merkezleme

## Gereksinimler

- Python 3.8 veya üzeri
- NumPy
- Matplotlib

## Kurulum

1. Projeyi klonlayın:
```bash
git clone [proje-url]
```

2. Gerekli paketleri yükleyin:
```bash
pip install numpy matplotlib
```

3. Uygulamayı çalıştırın:
```bash
python main.py
```

## Kullanım

1. Uygulama başlatıldığında iki panelli bir arayüz açılacaktır:
   - Sol panel: Çizim alanı
   - Sağ panel: Tahmin sonuçları

2. Çizim yaparken:
   - Mouse ile sol panele rakam çizin
   - Çizimi temizlemek için "Temizle" butonunu kullanın
   - Tahmini görmek için "Tahmin Et" butonuna tıklayın

3. İpuçları:
   - Rakamları net ve belirgin çizin
   - Çok ince çizgilerden kaçının
   - Rakamı çizim alanının ortasına yakın çizmeye çalışın

## Teknik Detaylar

- Model Mimarisi:
  - Giriş katmanı: 784 nöron (28x28 piksel)
  - Gizli katman: 20 nöron
  - Çıkış katmanı: 10 nöron (0-9 rakamları)
  
- Aktivasyon Fonksiyonu: Sigmoid
- Öğrenme Oranı: 0.01
- Eğitim Veri Seti: MNIST

## Nasıl Çalışır?

1. **Eğitim Aşaması:**
   - Program başlatıldığında, model MNIST veri seti üzerinde eğitilir
   - Her epoch sonunda doğruluk oranı gösterilir

2. **Tahmin Aşaması:**
   - Kullanıcının çizdiği rakam 28x28 piksellik bir görüntüye dönüştürülür
   - Görüntü otomatik olarak merkezlenir
   - Model görüntüyü işler ve en olası rakamı tahmin eder
   - Tahmin ve güven skoru ekranda gösterilir
