import werpy

class LMR_Metrics:
    @staticmethod
    def wer(reference_sentences, predicted_sentences):
        if len(reference_sentences) != len(predicted_sentences):
            raise ValueError("The number of reference sentences must match the number of predicted sentences.")

        wer_scores  = werpy.wers(reference_sentences, predicted_sentences)
        average_wer = sum(wer_scores) / len(wer_scores)
        return average_wer
    
    @staticmethod
    def wer_dict(reference_sentences, predicted_sentences):
        def extract(sentences):
            results = []
            for sentence in sentences:
                result = " ".join([word for d in sentence for word, tag in d.items() if tag != 'O'])
                if result == "":
                    result = " "
                results.append(result.strip())
            return results
    
        if len(reference_sentences) != len(predicted_sentences):
            raise ValueError("The number of reference sentences must match the number of predicted sentences.")

        # Extract entities from both reference and predicted sentences
        reference_entities = extract(reference_sentences)
        predicted_entities = extract(predicted_sentences)

        wer_scores  = werpy.wers(reference_sentences, predicted_sentences)
        average_wer = sum(wer_scores) / len(wer_scores)
        return average_wer
    
    def wer_type(reference_sentences, predicted_sentences):
        def extract(sentences):
            results = []
            for sentence in sentences:
                result = " ".join([word for sublist in sentence for word in sublist if word != 'O'])
                if result == "":
                    result = "Empty"
                results.append(result)
            return results

        if len(reference_sentences) != len(predicted_sentences):
            raise ValueError("The number of reference sentences must match the number of predicted sentences.")

        ref_entities = extract(reference_sentences)
        pred_entities = extract(predicted_sentences)

        wer_scores = werpy.wers(ref_entities, pred_entities)
        average_wer = sum(wer_scores) / len(wer_scores)
        return average_wer
