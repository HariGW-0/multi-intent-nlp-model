import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiIntentClassifier(nn.Module):
    def __init__(self, num_intents, model_name="bert-base-uncased", dropout_prob=0.3):
        super(MultiIntentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

def load_model_with_architecture(model_path, num_intents=10):
    """
    Load the model with the correct architecture
    """
    model = MultiIntentClassifier(num_intents=num_intents)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Example usage
if __name__ == "__main__":
    # This should be used with the reconstructed model
    model = load_model_with_architecture("multi_intent_model_reconstructed.pth", num_intents=10)
    print("✅ Model loaded with correct architecture!")
