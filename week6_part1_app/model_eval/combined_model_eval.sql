CREATE OR REPLACE TABLE
  `scg-l200-genai2.model_fine_tuning.model_evaluation` AS
SELECT
  "gemini-1.0-pro-002_tuned" AS evaluated_model,
  *
FROM
  ML.EVALUATE( MODEL `scg-l200-genai2.model_fine_tuning.gemini-pro-finetuned`,
    TABLE `scg-l200-genai2.model_fine_tuning.eval_cleaned`,
    STRUCT('TEXT_GENERATION' AS task_type,
      1024 AS max_output_tokens,
      0.1 AS temperature,
      1 AS top_k,
      0.3 AS top_p))
UNION ALL
SELECT
  "gemini-1.5-flash-002" AS evaluated_model,
  *
FROM
  ML.EVALUATE( MODEL `scg-l200-genai2.model_fine_tuning.gemini-flash`,
    TABLE `scg-l200-genai2.model_fine_tuning.eval_cleaned_flash`,
    STRUCT('TEXT_GENERATION' AS task_type,
      1024 AS max_output_tokens,
      0.1 AS temperature,
      1 AS top_k,
      0.3 AS top_p))
