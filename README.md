# ⚽ Serie A Predictor

Sistema di predizione partite Serie A basato su dati storici reali, modelli statistici (Poisson, Dixon-Coles, Elo) e fonti multiple gratuite.

## 🎯 Caratteristiche

- **Dati storici**: 30+ stagioni (1993/94 - oggi) da football-data.co.uk / datahub.io
- **Partite future**: Da football-data.org API e API-Football (free tier)
- **Statistiche avanzate**: xG, tiri, corner, carte da FBref (scraping)
- **Modelli ensemble**: Poisson + Dixon-Coles + Elo con pesi ottimizzati
- **Output**: Probabilità 1X2, Over/Under 2.5, BTTS, Risultati esatti
- **Dashboard**: Streamlit interattiva con value bets detection
- **Automazione**: GitHub Actions per aggiornamenti giornalieri gratuiti

## 📊 Fonti Dati (Tutte Gratuite)

| Fonte | Cosa fornisce | Limiti |
|-------|--------------|--------|
| football-data.co.uk | CSV storici con quote (1993-oggi) | Aggiornamento 2x/settimana |
| datahub.io | Mirror CSV con URL stabili | Stesso di sopra |
| football-data.org API | Partite future, classifiche, squadre | 10 req/min, free tier |
| API-Football | Partite future, predizioni, quote, stats | 100 req/giorno free |
| FBref (scraping) | xG, xA, passing, defense, possession | Rate limit ~5s/req |

## 🚀 Quick Start

### 1. Clone e Setup

```bash
cd seriea_predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configura API Keys (Opzionale ma Consigliato)

Crea file `.env`:
```bash
# football-data.org - registrati gratis su https://www.football-data.org/client/register
FOOTBALL_DATA_ORG_API_KEY=your_key_here

# API-Football - registrati gratis su https://www.api-football.com/
API_FOOTBALL_KEY=your_key_here
```

### 3. Inizializza Database e Scarica Dati Storici

```bash
python main.py init
```
*Scarica ~30 stagioni, crea database SQLite, ~2-3 minuti*

### 4. Lancia Dashboard

```bash
python main.py dashboard
```
*Apre http://localhost:8501*

## 📁 Struttura Progetto

```
seriea_predictor/
├── main.py                 # Entry point CLI
├── config.yaml             # Configurazione completa
├── requirements.txt        # Dipendenze Python
├── .env                    # API keys (non committare!)
├── data/
│   ├── seriea.db          # Database SQLite (auto-generato)
│   └── raw/               # CSV grezzi per debug
├── src/
│   ├── database.py        # Modelli SQLAlchemy
│   ├── fetchers/          # Data fetchers
│   │   ├── football_data_co_uk.py
│   │   ├── datahub_io.py
│   │   ├── api_football.py
│   │   └── fbref.py
│   ├── processing.py      # Merge, feature engineering, DB load
│   ├── models.py          # Poisson, Dixon-Coles, Elo, Ensemble
│   ├── scheduler.py       # Automazione APScheduler
│   └── dashboard.py       # Streamlit dashboard
├── .github/workflows/     # GitHub Actions per automazione
└── README.md
```

## 🛠 Comandi CLI

```bash
# Inizializzazione completa (prima volta)
python main.py init

# Aggiornamenti singoli
python main.py update --historical    # Dati storici (giornaliero)
python main.py update --upcoming      # Partite future (ogni 6h)
python main.py update --predictions   # Rigenera predizioni
python main.py update --live          # Live scores (ogni 15min match day)
python main.py update --retrain       # Riaddestra modelli (settimanale)

# Dashboard
python main.py dashboard --port 8501

# Scheduler continuo (blocca terminale)
python main.py scheduler

# Backtest su dati storici
python main.py backtest
```

## 📈 Modelli Implementati

### Poisson Model
- Stima attack/defense strength per squadra
- Home advantage factor
- Probabilità matrice scoreline

### Dixon-Coles
- Correzione rho per punteggi bassi (0-0, 1-0, 0-1, 1-1)
- Time decay (partite recenti peso maggiore)
- Ottimizzazione MLE con vincoli

### Elo Rating
- K-factor 20, home advantage 65 punti
- Rating iniziale 1500
- Aggiornamento cronologico

### Ensemble
- Pesi: Poisson 35%, Dixon-Coles 40%, Elo 25%
- Media pesata probabilità
- Scorelines dal modello migliore

## 🎯 Dashboard Features

- **Prossime Partite**: Probabilità 1X2, O/U 2.5, BTTS con grafici
- **Value Bets**: Confronto modello vs quote implicite (edge > 5%)
- **Classifica**: Calcolata da dati reali, colorata per zone
- **Analisi Squadra**: Forma ultime 5, prossime partite, xG
- **Aggiornamento manuale**: Pulsanti per trigger update/retrain

## ☁️ Deploy Gratuito

### Streamlit Cloud (Consigliato)
1. Push su GitHub
2. Connetti repo su https://share.streamlit.io/
3. Aggiungi secrets: `FOOTBALL_DATA_ORG_API_KEY`, `API_FOOTBALL_KEY`
4. Deploy automatico ad ogni push

### Railway / Render
```yaml
# railway.toml o render.yaml
build: pip install -r requirements.txt
start: python main.py dashboard --port $PORT
```

### GitHub Actions (Già configurato)
- `.github/workflows/daily-update.yml` fa tutto automaticamente
- Committa `data/seriea.db` aggiornato sul repo
- Dashboard su Streamlit Cloud legge DB aggiornato

## 🔧 Configurazione Avanzata

Modifica `config.yaml`:

```yaml
models:
  poisson:
    enabled: true
    min_matches_for_form: 6
  dixon_coles:
    enabled: true
    rho: 0.13
    time_decay: 0.001
  elo:
    enabled: true
    k_factor: 20

prediction:
  upcoming_matches_days: 14
  min_confidence_threshold: 0.15
```

## 📝 Note Importanti

1. **Solo scopi educativi/informativi** - Non usare per scommesse reali
2. **Rate limits**: Rispetta i limiti delle API gratuite
3. **FBref scraping**: Usa con moderazione, rispetta robots.txt
4. **Database**: SQLite va bene per uso singolo, per produzione usa PostgreSQL
5. **Modelli**: Sono baseline - per produzione serve feature engineering avanzato

## 🐛 Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Database locked | Chiudi dashboard prima di `main.py update` |
| API rate limit | Aspetta, riduci frequenza in config.yaml |
| FBref blocked | Imposta `use_selenium: true` in config, installa chromedriver |
| Nessuna predizione | Esegui `python main.py update --predictions` dopo init |

## 📚 Risorse Utili

- [football-data.co.uk](https://www.football-data.co.uk/) - CSV storici
- [football-data.org](https://www.football-data.org/) - API free
- [API-Football](https://www.api-football.com/) - API completa free tier
- [FBref](https://fbref.com/) - Statistiche avanzate
- [Dixon-Coles Paper](https://www.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf)
- [Elo Ratings](https://en.wikipedia.org/wiki/Elo_rating_system)

## 🤝 Contribuire

1. Fork del repo
2. Crea branch feature
3. Aggiungi test se possibile
4. Pull Request

## 📄 Licenza

MIT License - Vedi LICENSE per dettagli.

---

**⚠️ Disclaimer**: Questo strumento è solo a scopo educativo e informativo. Le predizioni non garantiscono risultati reali. Il gioco d'azzardo può causare dipendenza - gioca responsabilmente.