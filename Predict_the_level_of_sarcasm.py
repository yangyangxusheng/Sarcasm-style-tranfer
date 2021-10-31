import torch
import json
from run_pplm_discrim_train import predict,Discriminator
from pplm_classification_head import ClassificationHead

EPSILON = 1e-10
pretrained_model = 'gpt2-medium'
idx2class = ['0', '1']

def load_classifier_head(weights_path, meta_path, device):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size']
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params

def load_discriminator(weights_path, meta_path, device):
    classifier_head, meta_param = load_classifier_head(
        weights_path, meta_path, device
    )
    discriminator =  Discriminator(
        pretrained_model=meta_param['pretrained_model'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_param

discriminator, meta = load_discriminator(weights_path='headlines_sarcasm_discriminator(7).pt', meta_path='sarcasm_classifier_head_meta.json', device='cpu')
# sentence = 'the biggest fails of the first 100 years'

def predict_the_score_of_sentence(sentence):
    predict(sentence, discriminator, idx2class, cached=False, device='cpu')

# predict_the_score_of_sentence(sentence)