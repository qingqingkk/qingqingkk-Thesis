import torch.nn as nn
from transformers import AutoModelForAudioClassification,Wav2Vec2Model
import torch
import os

class Wav2Vec2SharedTransformerModel(nn.Module):
    def __init__(self, model_cs, 
                 model_sv,
                 shared_transformer_model="facebook/wav2vec2-base-960h",  # General Pre-trained Models
                 fusion_method='concat', hidden_size=768, num_classes=2):
        super(Wav2Vec2SharedTransformerModel, self).__init__()
        self.model_cs = model_cs
        self.model_sv = model_sv
        
        # load pretrained, only Transformer encoder
        self.shared_transformer = Wav2Vec2Model.from_pretrained(shared_transformer_model).encoder
        
        # Shared proj layer : map both modalities to the same vector space
        self.shared_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fusion_method = fusion_method
        
        # define fusion layer
        if self.fusion_method == 'attention':     
            self.atten_layer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
            
        elif self.fusion_method == 'concat':
            
            pass
        self.post_norm = nn.LayerNorm(hidden_size * 2)
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size) 
        self.dropout = nn.Dropout(p=0.1)
        
        # define classify head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_values_cs, input_values_sv):
        input_values_cs = input_values_cs.squeeze(1)  # Remove the unnecessary singleton dimension
        input_values_sv = input_values_sv.squeeze(1)
        
        # Freeze feature_encoder parameters
        self.model_cs.freeze_feature_encoder()
        self.model_sv.freeze_feature_encoder()
        
        # Extract features
        features_cs = self.model_cs.feature_extractor(input_values_cs)
        features_cs = features_cs.transpose(1, 2)  # adjust dimension
        hidden_states_cs, _ = self.model_cs.feature_projection(features_cs)

        features_sv = self.model_sv.feature_extractor(input_values_sv)
        features_sv = features_sv.transpose(1, 2)
        hidden_states_sv, _ = self.model_sv.feature_projection(features_sv)
       
        # Project to the same vector space
        projected_hidden_states_cs = self.layer_norm(self.shared_projection(hidden_states_cs))  # [batch_size, seq_len, hidden_size]
        projected_hidden_states_sv = self.layer_norm(self.shared_projection(hidden_states_sv))  # [batch_size, seq_len, hidden_size]
      
        # Fuse features

        if self.fusion_method == 'attention':
            # Transpose to adapt to the multi-head attention input format
            projected_hidden_states_cs = projected_hidden_states_cs.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
            projected_hidden_states_sv = projected_hidden_states_sv.transpose(0, 1)
            
            # Cross attention 1：'query', 'key', and 'value'
            attn_output1, _ = self.atten_layer(projected_hidden_states_cs, projected_hidden_states_sv, projected_hidden_states_sv)

            # Cross attention 2：'query', 'key', and 'value'
            attn_output2, _ = self.atten_layer(projected_hidden_states_sv, projected_hidden_states_cs, projected_hidden_states_cs)

            # Transform back and concatenate
            attn_output1 = attn_output1.transpose(0, 1)
            attn_output2 = attn_output2.transpose(0, 1)
            fused_features = torch.cat((attn_output1, attn_output2), dim=-1)  # [batch_size, seq_len, hidden_size * 2]
             

        elif self.fusion_method == 'concat':
            # Concatenate
            fused_features = torch.cat((projected_hidden_states_cs, projected_hidden_states_sv), dim=-1)
            
        # Pass through the fusion layer to reduce to hidden_size
        fused_features = self.post_norm(fused_features)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.dropout(fused_features) 

        
        # fused_features.to(device)
        # Put into Transformer encoder
        transformer_outputs = self.shared_transformer(fused_features)

        # Average pooling and classify
        pooled_features = transformer_outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_features)
        
        return logits


def load_model(args):
    if args.strategy == 'late':
        model1 = AutoModelForAudioClassification.from_pretrained(args.cp_path1, num_labels=args.num_classes)
        model2 = AutoModelForAudioClassification.from_pretrained(args.cp_path2, num_labels=args.num_classes)
        
        return [model1, model2]
        
    elif args.strategy == 'mid':
        model1 = AutoModelForAudioClassification.from_pretrained(args.cp_path1, num_labels=args.num_classes)
        model2 = AutoModelForAudioClassification.from_pretrained(args.cp_path2, num_labels=args.num_classes)
        
        model = Wav2Vec2SharedTransformerModel(model1, model2, fusion_method=args.mid_type, num_classes=args.num_classes)
    else:
       
        model = AutoModelForAudioClassification.from_pretrained(args.model_name, num_labels=args.num_classes)
    
    return model
