from bert_score import BERTScorer

def compute_F1_bertscore_corpus(y_preds: List, y_trues: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Compute the BERT score over a corpus (i.e. multiple sentences). Return values: P, R, F1, hash_value

    For each x,y pair (prediction and true translation), the function returns the precision (P), recall (R), F1 score and the hash_value (what model was used to make the BERT score)

    To get the system-level P, R or F1, use the torch mean() function, e.g. P.mean().

    Here, a "system" denotes a translation system. So system-level F1 score is the mean F1 score of all x,y pairs in the corpus.

    The intended use of this function is to pass all predictions and trues translations in a dataset filtered on language and translation model/system, e.g.:

    y_pred = df[(df["translation_model" == "MyModel"]) & df["CorrectLanguage"] == "dan"]["predictions"]

    args:

        y_preds: texts translated by translation model

        y_trues: the real/true translation

        language: three letter ISO language code of the y_preds and y_trues. Since Rabbla only translates to English, "eng" is the default
    
    """


    scorer = BERTScorer()

    (P, R, F1), hash_value = scorer.score(y_preds, y_trues, return_hash=True)

    #return P, R, F1, hash_value
    return F1


def get_sentence_level_scores(df):
    """
    Per sentence (row) in df, compute BERT score and Comet score
    """

    result = apply_bertscore(df)

    result2 = apply_cometscore(df)

    result3 = get_translationscore(df)

    df = pd.concat([df, result, result2, result3], axis=1)

    today = date.today()

    df["evaluation_date"] = today

    return df


def apply_bertscore(df):
    y_preds = df['translatedText'].tolist()
    y_trues = df['CorrectTranslation'].tolist()

    # Unpack the returned tensors
    P, R, F1, _ = compute_bertscore_corpus(y_preds, y_trues)

    # Return a DataFrame with the scores
    return pd.DataFrame({
        'BERT_P': P.numpy(),  # Convert tensor to numpy array for easy DataFrame compatibility
        'BERT_R': R.numpy(),
        'BERT_F1': F1.numpy()
    }, index=df.index)
