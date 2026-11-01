# Elma Konsantresi Tağşiş Tespiti (FTIR)

70 Brix elma konsantresinin FTIR-ATR spektrumundan tağşiş tespiti yapan, **kurulum gerektirmeyen** tarayıcı uygulaması.

**Uygulama:** GitHub Pages adresi (depo ayarlarında *Pages* bölümünde görünür). Sayfayı açıp spektrum dosyalarını sürüklemek yeterlidir; hesaplamaların tamamı tarayıcıda yapılır, dosyalar hiçbir sunucuya gönderilmez.

## Karar ağacı

```
spektrum
  └─ 1. Otantiklik (OC-SIMCA)            → otantik  ─────────────► sonuç: Otantik
        └─ tağşişli
             └─ 2. Şurup mu, meyve konsantresi mi?
                   ├─ şurup  → 3a. Glikoz mu, fruktoz mu?   → 4. o adulterantın regresyon modeli → oran (%)
                   └─ meyve  → 3b. Hurma mı, üzüm mü?      → 4. o adulterantın regresyon modeli → oran (%)
```

Ek kontroller:
- **Kapsam kontrolü:** spektrum eğitim verisindeki hiçbir örneğe benzemiyorsa uyarı verilir (ölçüm hatası, farklı ürün, tanımsız adulterant).
- **Düşük güven:** sınıflandırma skoru, çapraz doğrulamada doğruluğun %80'in altına düştüğü bölgedeyse işaretlenir.
- **Oran aralığı:** tahminle birlikte ±1,96·RMSEP (nested çapraz doğrulama) verilir.

## Desteklenen dosyalar

| Biçim | Açıklama |
|---|---|
| `.spa` | Thermo OMNIC tek spektrum |
| `.tsv` / `.csv` / `.txt` | İki sütun: dalga sayısı (cm⁻¹), absorbans. Başlık satırı, `;` `,` sekme ve ondalık virgül desteklenir |
| Matris `.csv` | İlk satır dalga sayıları, sonraki her satır bir örnek (ilk sütun isteğe bağlı örnek adı) |

Spektrum en az 850–1800 cm⁻¹ aralığını kapsamalıdır (seçilen modellere göre daha geniş olabilir).

## Modelleme

- **Veri:** 2025 örnekleri; otantik konsantreler ve %2, 4, 6, 10, 20, 30 oranlarında glikoz şurubu, fruktoz şurubu, hurma ve üzüm konsantresi karıştırılmış seriler.
- **Tarama:** Her düğüm için 4 spektral bölge × {yok, SNV, MSC, EMSC} × {yok, SG düzleştirme, SG 1./2./3. türev} ön işlemeleri ile DD-SIMCA, PLS / PLS-DA, PCA-LDA, ridge, RBF çekirdek (LS-SVM), kNN, MLP ve 1D-CNN kombinasyonları (görev başına binlerce aday).
- **Doğrulama:** Tağşişli örnekler aynı ebeveyn elmadan üretildiği için **leave-one-family-out**: bir partinin tüm örnekleri (ebeveyn dahil) birlikte dışarıda bırakılır. Model seçimi iç döngüde yapılır (nested LOFO). Uygulamada gösterilen performanslar bu nested değerlerdir.
- **Seçim:** 1-SE kuralı (en iyiye istatistiksel olarak eşdeğer adaylar içinde en basit model).
- **OC-SIMCA kabul sınırı:** eğitimde görülmemiş otantik örneklerin en az %95'i kabul edilecek şekilde.

## Sınırlamalar

- Düşük tağşiş oranları ve özellikle fruktoz şurubu FTIR ile zor ayırt edilir. "Otantik" sonucu tağşiş olmadığını kanıtlamaz; yalnızca bu yöntemin tespit sınırının altında olduğunu gösterir. Seviye bazında tespit oranları uygulamanın "Modeller ve doğrulama" bölümündedir.
- Modeller tek laboratuvarın 2025 örnekleri ve her adulterantın sınırlı sayıda lotuyla kurulmuştur. Farklı cihaz, bölge, çeşit ya da adulterant lotlarında performans düşebilir. İki adulterantın birlikte kullanıldığı karışımlar modellenmemiştir.
- Sonuçlar tarama amaçlıdır; resmi karar için doğrulayıcı analizlerle (ör. izotop oranı, oligosakkarit profili) desteklenmelidir.

## Depo yapısı

```
docs/          GitHub Pages: index.html, app.js, engine.js (hesap motoru), model.json (eğitilmiş modeller)
train/         Model eğitimi (Python, PyTorch-ROCm GPU)
  build_dataset.py   ham .spa/.tsv dosyalarından temiz veri seti
  search.py          nested LOFO ile ön işleme × teknik taraması (GPU)
  select_models.py   seçim, nested performans, uçtan uca değerlendirme, rapor
  export.py          seçilen modelleri docs/model.json'a yazar
  test_engine.js     tarayıcı motorunun Python ile birebir aynı sonuç verdiğini doğrular
```

Ham veriler ve ara sonuçlar depoya dahil değildir.

### Modelleri yeniden eğitmek

```bash
python train/build_dataset.py "<ham veri klasörü>" data
python train/search.py && python train/search.py --cnn-only
python train/select_models.py
python train/export.py
node train/test_engine.js
```
