# Elma Konsantresi Tağşiş Tespiti (FTIR)

70 Brix elma konsantresinin FTIR-ATR spektrumundan tağşiş tespiti yapan, **kurulum gerektirmeyen** tarayıcı uygulaması.

**Uygulama:** GitHub Pages adresi (depo ayarlarında *Pages* bölümünde görünür). Sayfayı açıp spektrum dosyalarını sürüklemek yeterlidir; hesaplamaların tamamı tarayıcıda yapılır, dosyalar hiçbir sunucuya gönderilmez.

## Karar ağacı

```
spektrum
  └─ 0. Ölçüm uygunluk kontrolü          → uygunsuz ─────────────► sonuç: Ölçüm reddedildi
  └─ 1. Otantiklik (OC-SIMCA, iki eşik)  → otantik  ─────────────► sonuç: Otantik
        └─ şüpheli (%80–%95 eşikleri arası) ya da tağşişli (%95 eşiği üstü)
             └─ 2. Şurup mu, meyve konsantresi mi?
                   ├─ şurup  → 3a. Glikoz mu, fruktoz mu?   → 4. o adulterantın regresyon modeli → oran (%)
                   └─ meyve  → 3b. Hurma mı, üzüm mü?      → 4. o adulterantın regresyon modeli → oran (%)
```

OC-SIMCA eşikleri: görülmemiş otantik örneklerin %95'inin kabul edildiği eşiğin üstü **tağşişli**; %80 ile %95 eşikleri arası **şüpheli**. Şüpheli örnekler de karar ağacından geçirilir ve *olası* tür/oran raporlanır (otantik örneklerin ~%15'i şüpheli çıkabilir; şüpheli sonuç hüküm değil, doğrulayıcı analiz önerisidir).

Ek kontroller:
- **Düşük güven:** sınıflandırma skoru, çapraz doğrulamada doğruluğun %80'in altına düştüğü bölgedeyse işaretlenir.
- **Oran aralığı:** tahminle birlikte ±1,96·RMSEP (nested çapraz doğrulama) verilir.

## Ölçüm uygunluk kontrolü

Analizden önce her spektrum kontrol edilir; **red** maddesi olan ölçümlerde sonuç verilmez (numune yeniden ölçülmelidir).

| Kontrol | Kaynak | Sonuç |
|---|---|---|
| Veri birimi (Absorbans), çözünürlük, örnek / arka plan tarama sayısı, ATR aksesuarı, arka plan yaşı, spektral aralık | `.spa` başlığı | red (tarama sayısı eğitimdekinden düşükse uyarı) |
| Cihaz seri numarası | `.spa` başlığı | uyarı (model başka cihazda doğrulanmamıştır) |
| Gürültü, parmak izi tepe yüksekliği (kristal teması), O–H bandı (Brix), baz çizgisi, negatif absorbans | spektrum | red / uyarı |
| Eğitim verisine benzerlik (DD-SIMCA, α = 0,001) | spektrum | red |

Kriterler eğitim verisinin ölçüm profilinden türetilir ve `model.json` içinde saklanır; model güncellendiğinde kriterler de güncellenir. `.tsv/.csv` dosyalarında ölçüm parametreleri okunamadığı için yalnızca spektral kontroller yapılır.

## Model güncelleme

- **Herkes için:** yeni `docs/model.json` dosyasını depoya gönderin; sayfa açıldığında yerleşik model olarak yüklenir.
- **Yalnızca kendi tarayıcınızda:** sayfadaki *Yeni model yükle (.json)* düğmesiyle; yüklenen modeller tarayıcıda saklanır, *Kayıtlı modeller* listesinden sürümler arasında geçiş yapılabilir veya yerleşik modele dönülebilir. Yüklenen dosya yapısal olarak doğrulanır ve örnek bir spektrumla sınanır.

## Rapor

*Raporu PDF olarak kaydet* düğmesi yazdırma penceresini açar (hedef: "PDF olarak kaydet"). Rapor; tarih, analist, not, model sürümü, özet tablo ve her numune için ölçüm parametreleri, karar yolu, kontrol sonuçları ve uyarıları içerir. *CSV indir* aynı bilgileri tablo olarak verir.

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
docs/          GitHub Pages: index.html, app.js, engine.js (hesap motoru), model.json (eğitilmiş modeller + kabul kriterleri)
train/         Model eğitimi (Python, PyTorch-ROCm GPU)
  build_dataset.py   ham .spa/.tsv dosyalarından temiz veri seti (+ .spa ölçüm parametreleri)
  spa_meta.py        .spa başlığından ölçüm parametrelerini okur
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
