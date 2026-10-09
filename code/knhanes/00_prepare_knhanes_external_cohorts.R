# -*- coding: UTF-8 -*-
options(stringsAsFactors = FALSE)
invisible(Sys.setlocale("LC_ALL", "English_United States.utf8"))

suppressPackageStartupMessages({
  library(haven)
  library(gamlss)
  library(gamlss.dist)
})

source_root <- Sys.getenv("PAPER3_KNHANES_SOURCE_ROOT")
repo_root <- normalizePath(file.path(dirname(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]), "..", ".."), mustWork = FALSE)
output_root <- Sys.getenv("PAPER3_PRIVATE_DATA_ROOT", file.path(repo_root, "data_private"))
scorer_root <- Sys.getenv("PAPER1_ALM_SCORER_ROOT")
if (source_root == "") stop("Set PAPER3_KNHANES_SOURCE_ROOT to your authorized local KNHANES source directory.")
if (scorer_root == "") stop("Set PAPER1_ALM_SCORER_ROOT to the local nhanes-alm-reference-scorer directory.")

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_root, "logs"), recursive = TRUE, showWarnings = FALSE)

source(file.path(scorer_root, "R", "score_conditional_alm.R"), local = .GlobalEnv)
bundles <- load_alm_reference(file.path(scorer_root, "models", c(
  "corrected_H_18_69_Female_bundles.rds",
  "corrected_H_18_69_Male_bundles.rds"
)))

clean_numeric <- function(x) suppressWarnings(as.numeric(zap_labels(zap_missing(x))))
valid_waist <- function(x) is.finite(x) & x >= 40 & x <= 200

score_z <- function(id, sex, age, height_cm, alm_kg) {
  input <- data.frame(row_key = seq_along(id), sex = sex, age = age,
                      height_m = height_cm / 100, alm_kg = alm_kg)
  scored <- score_conditional_alm(input, bundles = bundles, warn = FALSE)$pooled
  out <- rep(NA_real_, length(id))
  out[scored$row_key] <- scored$model_averaged_z
  out
}

make_targets <- function(x, alm, suffix) {
  z <- score_z(x$id, x$sex_label, x$age, x$height_cm, alm)
  almi <- alm / (x$height_cm / 100)^2
  alm_bmi <- alm / x$bmi
  x[[paste0("z_", suffix)]] <- z
  x[[paste0("p5_", suffix)]] <- as.integer(z < qnorm(.05))
  x[[paste0("ewgsop2_", suffix)]] <- as.integer(ifelse(x$sex_label == "Male", almi < 7.0, almi < 5.5))
  x[[paste0("fnih_", suffix)]] <- as.integer(ifelse(x$sex_label == "Male", alm_bmi < 0.789, alm_bmi < 0.512))
  x
}

# Primary KNHANES 2008-2011 cohort: same Paper 3 age range, raw Korean ALM primary.
historical <- readRDS(file.path(source_root, "data", "knhanes_2008_2011_analysis_19_69_official_weight.rds"))
waist_rows <- list()
raw_names <- c(`2008` = "HN08_ALL.sav", `2009` = "HN09_all.sav", `2010` = "HN10_all.sav", `2011` = "HN11_all.sav")
for (yr in names(raw_names)) {
  path <- file.path(source_root, "raw_all_sav", raw_names[[yr]])
  header <- read_sav(path, n_max = 1)
  selected <- names(header)[tolower(names(header)) %in% c("id", "he_wc")]
  z <- as.data.frame(read_sav(path, col_select = all_of(selected)))
  names(z) <- tolower(names(z))
  z$id <- trimws(as.character(z$id))
  z$waist_cm <- clean_numeric(z$he_wc)
  z$waist_cm[!valid_waist(z$waist_cm)] <- NA_real_
  z$year <- as.integer(yr)
  waist_rows[[yr]] <- z[, c("id", "year", "waist_cm")]
}
waist_historical <- do.call(rbind, waist_rows)
stopifnot(!anyDuplicated(waist_historical[, c("id", "year")]))
historical <- merge(historical, waist_historical, by = c("id", "year"), all.x = TRUE, sort = FALSE)
historical <- historical[historical$age >= 20 & historical$age <= 59, ]
historical$sex_male <- as.integer(historical$sex_label == "Male")
historical <- make_targets(historical, historical$alm_raw_kg, "raw")
historical <- make_targets(historical, historical$alm_raw_kg * 0.946, "calibrated_0946")
historical$cohort <- "KNHANES 2008-2011"
historical$survey_weight <- historical$wt_pool

keep_historical <- c("id", "cohort", "year", "age", "sex_label", "sex_male", "height_cm", "weight_kg",
  "waist_cm", "bmi", "alm_raw_kg", "survey_weight", "strata_pool", "psu_pool",
  "z_raw", "p5_raw", "ewgsop2_raw", "fnih_raw", "z_calibrated_0946", "p5_calibrated_0946",
  "ewgsop2_calibrated_0946", "fnih_calibrated_0946")
historical <- historical[complete.cases(historical[, c("age", "sex_male", "height_cm", "weight_kg", "waist_cm",
  "survey_weight", "strata_pool", "psu_pool", "p5_raw", "ewgsop2_raw", "fnih_raw")]), keep_historical]

# Secondary contemporary cohort: KNHANES 2024 DXA is only available in the archived 40-69 analytic file.
contemporary <- readRDS(file.path(source_root, "data", "knhanes_2024_analysis_40_69.rds"))
path24 <- file.path(source_root, "raw_all_sav", "HN24_ALL.sav")
header24 <- read_sav(path24, n_max = 1)
selected24 <- names(header24)[tolower(names(header24)) %in% c("id", "he_wc")]
waist24 <- as.data.frame(read_sav(path24, col_select = all_of(selected24)))
names(waist24) <- tolower(names(waist24))
waist24$id <- trimws(as.character(waist24$id))
waist24$waist_cm <- clean_numeric(waist24$he_wc)
waist24$waist_cm[!valid_waist(waist24$waist_cm)] <- NA_real_
stopifnot(!anyDuplicated(waist24$id))
contemporary$id <- trimws(as.character(contemporary$id))
contemporary <- merge(contemporary, waist24[, c("id", "waist_cm")], by = "id", all.x = TRUE, sort = FALSE)
contemporary <- contemporary[contemporary$age >= 40 & contemporary$age <= 59, ]
contemporary$sex_male <- as.integer(contemporary$sex_label == "Male")
contemporary$alm_raw_kg <- contemporary$alm_kg
contemporary <- make_targets(contemporary, contemporary$alm_raw_kg, "raw")
contemporary <- make_targets(contemporary, contemporary$alm_raw_kg * 0.946, "calibrated_0946")
contemporary$cohort <- "KNHANES 2024"
contemporary$year <- 2024L
contemporary$survey_weight <- contemporary$wt
contemporary$strata_pool <- interaction(contemporary$year, contemporary$kstrata, drop = TRUE)
contemporary$psu_pool <- interaction(contemporary$year, contemporary$kstrata, contemporary$psu, drop = TRUE)
keep_contemporary <- keep_historical
contemporary <- contemporary[complete.cases(contemporary[, c("age", "sex_male", "height_cm", "weight_kg", "waist_cm",
  "survey_weight", "strata_pool", "psu_pool", "p5_raw", "ewgsop2_raw", "fnih_raw")]), keep_contemporary]

write.csv(historical, gzfile(file.path(output_root, "knhanes_2008_2011_paper3_ready.csv.gz")), row.names = FALSE)
write.csv(contemporary, gzfile(file.path(output_root, "knhanes_2024_paper3_ready.csv.gz")), row.names = FALSE)

audit <- data.frame(
  cohort = c("KNHANES 2008-2011", "KNHANES 2024"),
  participants = c(nrow(historical), nrow(contemporary)),
  age_min = c(min(historical$age), min(contemporary$age)),
  age_max = c(max(historical$age), max(contemporary$age)),
  missing_required = c(sum(!complete.cases(historical)), sum(!complete.cases(contemporary))),
  strata = c(length(unique(historical$strata_pool)), length(unique(contemporary$strata_pool))),
  psus = c(length(unique(historical$psu_pool)), length(unique(contemporary$psu_pool)))
)
write.csv(audit, file.path(output_root, "cohort_preparation_audit.csv"), row.names = FALSE)
print(audit, row.names = FALSE)
