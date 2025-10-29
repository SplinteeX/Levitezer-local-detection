# Käyttöönotto

## Lataa python versio 
Vähintään versio 3.8 ja enintään versio 3.11.9. Ultralytics ei tue kunnolla versiota 3.12 tai uudempia.

## Lataa venv ja vaadittavat paketit
Seuraavalla komennolla saat venv ja projektiin vaaditut paketit.
Suorita seuraavat komennot projektin terminaalissa ```python3 -m venv .venv``` ja ```pip install -r requirements.txt```

## Lataa valmiiksi koulutettu weight malli tai kouluta itse omasi.
Levitezer discordissa weight valmiiksi koulutettu weight malli.
Koulutusmateriaalit löytyvät Levitezer discord palvelimelta (Train, valid & test kansiot)

## Yhdistä malli api_server.py kanssa
Muokkaa seuraavaa api_server.py sisällä ```root_best = APP_BASE / "best.pt"``` varmista että tähän on liitetty sinun mallisi.

## Jos haluat pyörittää palvelinta lokaalisti netin sisällä (Tunnistus pyörii toisella laitteella)
Suorita seuraava komento adminina powershellissä: ```New-NetFirewallRule -DisplayName "YOLO API (8000)" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow``` - Huom tee tämä vain jos haluat sen pyörivän toisella laitteella. Vaatii saman wifi verkon.

## Käynnistus
Sovellus käynnistuu terminaalista seuraavalla komennolla: ```uvicorn api_server:app --host 0.0.0.0 --port 8000```

# Lokaali Droonien Tunnistus
Tämä projekti toteuttaa lokaalin droonien tunnistuksen käyttäen **YOLOv11**-mallia ja **FastAPI**-palvelinta. API mahdollistaa kuvien lähettämisen ja palauttaa tunnistetut dronet sekä annotoidun kuvan.

---

## 📁 Datan valmistelu
- Datasetti haettiin **Roboflow-palvelusta**.
- Lataamalla **train**, **test** ja **valid** -kansiot saadaan YOLO-mallin koulutukseen tarvittavat tiedostot.

---

## 🏋️‍♂️ Mallin koulutus
1. Loin tiedoston `train_yolov11.py`, joka sisältää mallin koulutuslogiikan.
2. Ensimmäinen kokeilumalli oli pieni nopean testauksen mahdollistamiseksi.
3. Treenaasin mallin roboflow datasetillä ja testasin sen toimintaa `api_server.py` kautta.

✅ Tulokset: Malli tunnistaa dronet suhteellisen hyvin ja suorituskyky on ~100 ms per kuva.

---

## ⚡ API-palvelin (`api_server.py`)

**Käyttötarkoitus:** Vastaanottaa kuvia ja suorittaa YOLOv11-pohjainen dronen tunnistus.

### Pääominaisuudet
- **POST /detect**
  - Tukee:
    - Multipart-lähetystä (`image`-kenttä)
    - JSON-muodossa base64-kuvaa
    - Kuva URL:n kautta
  - Palauttaa:
    - Tunnistetut dronet ja niiden koordinaatit
    - Annotoidun kuvan polun

- **Käynnistyessä:**  
  - Lataa YOLOv11-painotiedoston
  - Lukee `data.yaml` -tiedoston luokkanimet  
  - Tulostaa LAN-osoitteen API:n testaamiseen - api_server.py toimii lokaalisti internetin sisällä.

### Konfiguroitavat parametrit
```python
IMGSZ = 640  # Kuvan koko mallin syötteelle
CONF = 0.25  # Luottamuskynnys
DEVICE = None  # Laitevalinta (CPU/GPU)

