-- =============================================================
-- Agentic Data Preprocessing — SQLite Veritabanı Şeması
-- Proje: Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip
--        Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi
-- =============================================================
-- Çalıştırma: sqlite3 agentic_preprocessing.db < schema.sql
-- Gereksinim: SQLite 3.25.0+
-- Zaman dilimi: Tüm tarih/saat alanları UTC ISO8601 formatında
-- =============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode  = WAL;
PRAGMA encoding      = 'UTF-8';

-- =============================================================
-- TABLO: users
-- Amaç: Kimlik doğrulaması yapan kullanıcı hesapları.
-- =============================================================
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);


-- =============================================================
-- TABLO: datasets
-- Amaç: OpenML, UCI veya kullanıcı yüklemesi yoluyla sisteme
--       alınan veri setlerinin kaydı. Veri seti içerikleri
--       dosya sisteminde tutulur; veritabanında yalnızca meta
--       veri saklanır.
-- =============================================================
CREATE TABLE IF NOT EXISTS datasets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source            TEXT    NOT NULL
                              CHECK (source IN ('openml', 'uci', 'upload')),
    source_dataset_id TEXT,
    name              TEXT    NOT NULL,
    file_path         TEXT    NOT NULL UNIQUE,
    file_hash         TEXT    NOT NULL UNIQUE,   -- SHA-256; tekrar yüklemeyi önler
    row_count         INTEGER CHECK (row_count IS NULL OR row_count > 0),
    column_count      INTEGER CHECK (column_count IS NULL OR column_count > 0),
    file_size_bytes   INTEGER CHECK (file_size_bytes IS NULL OR file_size_bytes > 0),
    download_status   TEXT    NOT NULL DEFAULT 'pending'
                              CHECK (download_status IN
                                     ('pending', 'downloading', 'ready', 'failed', 'invalid')),
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_datasets_source ON datasets (source, source_dataset_id);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets (download_status);
CREATE INDEX IF NOT EXISTS idx_datasets_hash   ON datasets (file_hash);


-- =============================================================
-- TABLO: dataset_profiles
-- Amaç: Veri setinin kalıcı profil bilgisi. Profil, ilk
--       analizde eğitim verisi üzerinde hesaplanır ve veri
--       setine bağlı olarak tek sefer saklanır. Benzer veri
--       setlerini bulmak için profile_hash kullanılır.
-- Kısıt: Her veri setinin en fazla bir profili olabilir
--        (UNIQUE dataset_id).
-- =============================================================
CREATE TABLE IF NOT EXISTS dataset_profiles (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id               INTEGER NOT NULL UNIQUE
                                     REFERENCES datasets (id),
    numeric_column_count     INTEGER NOT NULL CHECK (numeric_column_count >= 0),
    categorical_column_count INTEGER NOT NULL CHECK (categorical_column_count >= 0),
    total_missing_pct        REAL    NOT NULL
                                     CHECK (total_missing_pct BETWEEN 0.0 AND 100.0),
    -- JSON: {"col_name": missing_pct, ...}
    missing_value_detail     TEXT,
    target_column            TEXT    NOT NULL,
    problem_type             TEXT    NOT NULL
                                     CHECK (problem_type IN ('classification', 'regression')),
    -- Yalnızca sınıflandırma problemlerinde doldurulur
    class_count              INTEGER CHECK (class_count IS NULL OR class_count >= 2),
    -- JSON: {"class_label": row_count, ...}
    class_balance_detail     TEXT,
    train_row_count          INTEGER CHECK (train_row_count IS NULL OR train_row_count > 0),
    test_row_count           INTEGER CHECK (test_row_count  IS NULL OR test_row_count  > 0),
    -- JSON: {"col_name": {"mean": ..., "std": ..., "min": ..., "max": ..., "median": ...}, ...}
    numeric_stats            TEXT,
    -- JSON: {"col_name": {"unique_count": ..., "rare_category_pct": ...}, ...}
    categorical_stats        TEXT,
    -- JSON: {"col_name": skewness_value, ...}
    skewness_detail          TEXT,
    -- Benzerlik araması için profil özeti hash değeri
    profile_hash             TEXT    NOT NULL,
    profiled_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_profiles_hash         ON dataset_profiles (profile_hash);
CREATE INDEX IF NOT EXISTS idx_profiles_problem_type ON dataset_profiles (problem_type);


-- =============================================================
-- TABLO: analysis_runs
-- Amaç: Kullanıcının başlattığı her analiz çalıştırması.
--       A/B/C/D deney modunu, rastgelelik tohumunu ve
--       deneme sınırını içerir.
-- =============================================================
CREATE TABLE IF NOT EXISTS analysis_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users    (id),
    dataset_id      INTEGER NOT NULL REFERENCES datasets (id),
    user_request    TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                            CHECK (status IN
                                   ('pending', 'running', 'completed', 'failed', 'cancelled')),
    -- Deney modu: A=sabit, B=tek seçim, C=geri bildirimli, D=bellek destekli
    experiment_mode TEXT    NOT NULL DEFAULT 'D'
                            CHECK (experiment_mode IN ('A', 'B', 'C', 'D')),
    random_seed     INTEGER NOT NULL DEFAULT 42,
    max_attempts    INTEGER NOT NULL DEFAULT 5
                            CHECK (max_attempts BETWEEN 1 AND 10),
    attempt_count   INTEGER NOT NULL DEFAULT 0
                            CHECK (attempt_count >= 0),
    started_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    error_message   TEXT,
    CHECK (attempt_count <= max_attempts)
);

CREATE INDEX IF NOT EXISTS idx_runs_user    ON analysis_runs (user_id);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON analysis_runs (dataset_id);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON analysis_runs (status);


-- =============================================================
-- TABLO: preprocessing_strategies
-- Amaç: Denenmiş veya tanımlanmış ön işleme stratejilerinin
--       global kataloğu. strategy_signature sayesinde aynı
--       strateji birden fazla kez eklenmez.
-- Not:  Bu tablo global paylaşımlıdır; bir stratejinin hangi
--       analizde denendiği evaluation_attempts üzerinden
--       izlenir.
-- =============================================================
CREATE TABLE IF NOT EXISTS preprocessing_strategies (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    imputation_numeric       TEXT    NOT NULL
                                     CHECK (imputation_numeric IN
                                            ('mean', 'median', 'knn', 'iterative', 'constant')),
    imputation_categorical   TEXT    NOT NULL
                                     CHECK (imputation_categorical IN
                                            ('most_frequent', 'constant', 'knn')),
    scaling_method           TEXT    NOT NULL
                                     CHECK (scaling_method IN
                                            ('standard', 'robust', 'minmax', 'none')),
    encoding_method          TEXT    NOT NULL
                                     CHECK (encoding_method IN
                                            ('onehot', 'ordinal', 'target', 'none')),
    outlier_handling         TEXT    NOT NULL DEFAULT 'none'
                                     CHECK (outlier_handling IN ('clip', 'remove', 'none')),
    feature_selection        TEXT    NOT NULL DEFAULT 'none'
                                     CHECK (feature_selection IN
                                            ('none', 'variance_threshold', 'selectkbest')),
    -- JSON: {"knn_neighbors": 5, "selectkbest_k": 10, ...}
    extra_params             TEXT,
    -- Yukarıdaki alanların deterministik hash'i; tekrarı önler
    strategy_signature       TEXT    NOT NULL UNIQUE,
    created_at               TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_strategies_signature ON preprocessing_strategies (strategy_signature);


-- =============================================================
-- TABLO: evaluation_attempts
-- Amaç: Bir analiz çalıştırması içindeki her değerlendirme
--       denemesinin kaydı. CV skoru, süre, LLM çağrı sayısı
--       ve hata bilgisi saklanır.
-- Kısıt 1: Aynı analizde aynı deneme numarası tekrar edilemez
--           → UNIQUE (analysis_run_id, attempt_number)
-- Kısıt 2: Aynı analizde aynı strateji tekrar denenemez
--           → UNIQUE (analysis_run_id, strategy_id)
-- =============================================================
CREATE TABLE IF NOT EXISTS evaluation_attempts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id      INTEGER NOT NULL REFERENCES analysis_runs           (id),
    strategy_id          INTEGER NOT NULL REFERENCES preprocessing_strategies (id),
    attempt_number       INTEGER NOT NULL CHECK (attempt_number >= 1),
    model_type           TEXT    NOT NULL
                                 CHECK (model_type IN
                                        ('logistic_regression', 'ridge_regression')),
    primary_metric_name  TEXT    NOT NULL
                                 CHECK (primary_metric_name IN
                                        ('macro_f1', 'balanced_accuracy', 'rmse', 'mae')),
    -- 5 katlı çapraz doğrulama ortalaması
    cv_primary_score     REAL,
    -- İkincil metrik: sınıflandırma → balanced_accuracy, regresyon → mae
    cv_secondary_score   REAL,
    -- JSON: {"fold_scores": [...], "mean": ..., "std": ...}
    cv_scores_detail     TEXT,
    duration_seconds     REAL    CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
    llm_call_count       INTEGER NOT NULL DEFAULT 0 CHECK (llm_call_count >= 0),
    status               TEXT    NOT NULL DEFAULT 'running'
                                 CHECK (status IN
                                        ('running', 'success', 'failed', 'invalid', 'skipped')),
    error_message        TEXT,
    attempted_at         TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE (analysis_run_id, attempt_number),
    UNIQUE (analysis_run_id, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_attempts_run      ON evaluation_attempts (analysis_run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_strategy ON evaluation_attempts (strategy_id);
CREATE INDEX IF NOT EXISTS idx_attempts_status   ON evaluation_attempts (status);


-- =============================================================
-- TABLO: experience_memory
-- Amaç: Projenin araştırma çekirdeği. Farklı veri setlerinde
--       denenen stratejilerin toplulaştırılmış performansını
--       saklar. profile_hash üzerinden benzer veri setleri
--       için strateji önerileri üretilir.
-- Kısıt: (profile_hash, strategy_id, problem_type) üçlüsü
--        benzersizdir — bir kombinasyon için tek satır.
-- Not:   Bu tablo Python katmanı tarafından güncellenir;
--        her başarılı/başarısız denemeden sonra ilgili satır
--        UPSERT ile yenilenir.
-- =============================================================
CREATE TABLE IF NOT EXISTS experience_memory (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    -- dataset_profiles.profile_hash ile örtüşür; strict FK yok,
    -- benzerlik araması Python'da yapılır
    profile_hash         TEXT    NOT NULL,
    strategy_id          INTEGER NOT NULL REFERENCES preprocessing_strategies (id),
    problem_type         TEXT    NOT NULL
                                 CHECK (problem_type IN ('classification', 'regression')),
    cv_score_avg         REAL    NOT NULL,
    cv_score_best        REAL    NOT NULL,
    attempt_count        INTEGER NOT NULL DEFAULT 1  CHECK (attempt_count >= 1),
    success_count        INTEGER NOT NULL DEFAULT 0  CHECK (success_count >= 0),
    avg_duration_seconds REAL    CHECK (avg_duration_seconds IS NULL OR avg_duration_seconds >= 0),
    -- Denetim için son güncellemeye katkıda bulunan kaynaklar
    last_dataset_id      INTEGER REFERENCES datasets      (id),
    last_analysis_run_id INTEGER REFERENCES analysis_runs (id),
    last_updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE (profile_hash, strategy_id, problem_type),
    CHECK (success_count <= attempt_count)
);

CREATE INDEX IF NOT EXISTS idx_memory_profile ON experience_memory (profile_hash, problem_type);
CREATE INDEX IF NOT EXISTS idx_memory_score   ON experience_memory (cv_score_avg DESC);


-- =============================================================
-- TABLO: analysis_reports
-- Amaç: Bir analiz çalıştırmasının nihai raporu. Test seti
--       sonuçlarını ve CV sonuçlarını birbirinden ayrı saklar;
--       test seti yalnızca burada görünür, strateji seçimine
--       dahil edilmez.
-- Kısıt: Her analiz çalıştırmasının en fazla bir raporu
--        olabilir (UNIQUE analysis_run_id).
-- =============================================================
CREATE TABLE IF NOT EXISTS analysis_reports (
    id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id            INTEGER NOT NULL UNIQUE
                                       REFERENCES analysis_runs           (id),
    best_strategy_id           INTEGER REFERENCES preprocessing_strategies (id),
    -- Nihai test seti metrikleri — strateji seçiminden sonra bir kez hesaplanır
    final_test_primary_score   REAL,
    final_test_secondary_score REAL,
    -- JSON: {"metric_name": value, ...}
    final_test_metrics_detail  TEXT,
    -- En iyi denemeden gelen CV skoru (referans karşılaştırması için)
    cv_primary_score           REAL,
    total_attempts             INTEGER NOT NULL DEFAULT 0
                                       CHECK (total_attempts >= 0),
    total_duration_seconds     REAL    CHECK (total_duration_seconds IS NULL
                                              OR total_duration_seconds >= 0),
    total_llm_calls            INTEGER NOT NULL DEFAULT 0
                                       CHECK (total_llm_calls >= 0),
    -- LLM tarafından üretilen strateji seçim gerekçesi
    selection_rationale        TEXT,
    report_created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reports_run ON analysis_reports (analysis_run_id);


-- =============================================================
-- VIEW: v_strategy_performance
-- Amaç: Deneyim belleği ile strateji detaylarını birleştirir.
--       Benzer profil için strateji önerisinde hızlı sorgulama
--       sağlar.
-- =============================================================
CREATE VIEW IF NOT EXISTS v_strategy_performance AS
SELECT
    em.id                    AS memory_id,
    em.profile_hash,
    em.problem_type,
    ps.id                    AS strategy_id,
    ps.strategy_signature,
    ps.imputation_numeric,
    ps.imputation_categorical,
    ps.scaling_method,
    ps.encoding_method,
    ps.outlier_handling,
    ps.feature_selection,
    em.cv_score_avg,
    em.cv_score_best,
    em.attempt_count,
    em.success_count,
    CASE
        WHEN em.attempt_count > 0
        THEN ROUND(CAST(em.success_count AS REAL) / em.attempt_count, 4)
        ELSE 0.0
    END                      AS success_rate,
    em.avg_duration_seconds,
    em.last_updated_at
FROM experience_memory em
JOIN preprocessing_strategies ps ON ps.id = em.strategy_id;
