-- CREATE OR REPLACE MODEL `model_fine_tuning.text_embedding_003`
--   REMOTE WITH CONNECTION `us-west1.gemini-fine-tuning`
--   OPTIONS(ENDPOINT = 'textembedding-gecko@003');

CREATE TEMP TABLE embed_hold AS
SELECT
  id,
  hotel_name,
  hotel_address,
  hotel_description,
  nearest_attractions,
  "textembedding-gecko@003" AS embedding_model,
  ml_generate_embedding_result AS nearest_attractions_embeddings,
  ml_generate_embedding_statistics.token_count AS nearest_attractions_token_count,
  ml_generate_embedding_statistics.truncated AS nearest_attractions_embeddings_truncated,
  CONCAT("[",ARRAY_TO_STRING(ARRAY(
      SELECT
        CAST(num AS STRING)
      FROM
        UNNEST(ml_generate_embedding_result) AS num), ", ", ""),"]") AS nearest_attractions_embeddings_string
FROM
  ML.GENERATE_EMBEDDING( MODEL `model_fine_tuning.text_embedding_004`,
    (
    SELECT
      *,
      nearest_attractions AS content
    FROM
      `scg-l200-genai2.hotels.florence`),
    STRUCT(TRUE AS flatten_json_output) );
CREATE OR REPLACE TABLE
  `hotels.florence_embeddings` AS
SELECT
  * EXCEPT(content,
    ml_generate_embedding_result,
    ml_generate_embedding_statistics,
    ml_generate_embedding_status),
  ml_generate_embedding_result AS hotel_description_embeddings,
  ml_generate_embedding_statistics.token_count AS hotel_description_token_count,
  ml_generate_embedding_statistics.truncated AS hotel_description_embeddings_truncated,

  CONCAT("[",ARRAY_TO_STRING(ARRAY(
      SELECT
        CAST(num AS STRING)
      FROM
        UNNEST(ml_generate_embedding_result) AS num), ", ", ""),"]") AS hotel_description_embeddings_string
FROM
  ML.GENERATE_EMBEDDING( MODEL `model_fine_tuning.text_embedding_004`,
    (
    SELECT
      *,
      hotel_description AS content
    FROM
      embed_hold),
    STRUCT(TRUE AS flatten_json_output) )
