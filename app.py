
import gradio as gr
import matplotlib.pyplot as plt
import io
import base64
import shap
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import re
import json
import pickle # Added for loading .pkl and .json files
import os # Added for path checking

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder # Explicitly import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture (MultiTaskMLP) ---
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(MultiTaskMLP, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        shared_features = self.shared_layers(x)
        ddg_output = self.regression_head(shared_features)
        effect_output = self.classification_head(shared_features)
        return ddg_output, effect_output

# --- SHAP Wrapper Classes ---
class RegressionHeadWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        ddg_output, _ = self.original_model(x)
        return ddg_output

class ClassificationHeadWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        _, effect_output = self.original_model(x)
        return effect_output

# --- Helper Function: plot_to_base64 ---
def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return f"<img src='data:image/png;base64,{image_base64}'>"

# --- Helper Function: AA_PROPERTIES and get_aa_property_comparison ---
# AA_PROPERTIES is now loaded from global_vars.json in the main script
def get_aa_property_comparison(original_aa, mutant_aa):
    # global AA_PROPERTIES is assumed to be loaded after global_vars.json
    if original_aa not in AA_PROPERTIES or mutant_aa not in AA_PROPERTIES:
        return "Invalid amino acid code(s) for comparison."
    results = {}
    for prop, _ in AA_PROPERTIES['A'].items():
        orig_val = AA_PROPERTIES[original_aa].get(prop, 0)
        mut_val = AA_PROPERTIES[mutant_aa].get(prop, 0)
        change = mut_val - orig_val
        results[prop] = {
            'original': orig_val,
            'mutant': mut_val,
            'change': f'{change:.2f}'
        }
    return results

# --- Helper Function: get_top_features_by_shap ---
def get_top_features_by_shap(explainer_obj, background_data_tensor, feature_names, top_n=10, task_type='regression'):
    if task_type == 'regression':
        shap_values_background = explainer_obj.shap_values(background_data_tensor)
        if isinstance(shap_values_background, list):
            shap_values_background = np.array(shap_values_background).squeeze()
        elif shap_values_background.ndim == 3:
            shap_values_background = shap_values_background.squeeze(axis=-1)
    elif task_type == 'classification':
        shap_values_background_raw = explainer_obj.shap_values(background_data_tensor)
        if isinstance(shap_values_background_raw, list):
            abs_shap_values = np.mean(np.abs(np.array(shap_values_background_raw)), axis=0)
        elif shap_values_background_raw.ndim == 3:
            abs_shap_values = np.mean(np.abs(shap_values_background_raw), axis=-1)
        else:
            abs_shap_values = np.abs(shap_values_background_raw)
        shap_values_background = abs_shap_values
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")
    mean_abs_shap = np.mean(np.abs(shap_values_background), axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    top_features = feature_importance.sort_values(by='Mean_Abs_SHAP', ascending=False).head(top_n)
    return top_features.to_dict(orient='records')

# --- Helper Function: plot_local_feature_importance ---
def plot_local_feature_importance(shap_values, feature_names, title="Local Feature Importance", top_n=10):
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': shap_values
    })
    feature_importance['Abs_SHAP'] = np.abs(feature_importance['SHAP_Value'])
    feature_importance = feature_importance.sort_values(by='Abs_SHAP', ascending=False).head(top_n)
    feature_importance = feature_importance.sort_values(by='SHAP_Value', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'blue' for x in feature_importance['SHAP_Value']]
    ax.barh(feature_importance['Feature'], feature_importance['SHAP_Value'], color=colors)
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    fig.tight_layout()
    return plot_to_base64(fig)

# --- Helper Function: get_mutation_context_summary ---
def get_mutation_context_summary(input_data_df):
    summary_html = "<h3>Mutation Structural Context:</h3>"
    sst = input_data_df['sst'].iloc[0] if 'sst' in input_data_df.columns else 'N/A'
    rsa_val = input_data_df['rsa'].iloc[0] if 'rsa' in input_data_df.columns else 'N/A'
    res_depth_val = input_data_df['res_depth'].iloc[0] if 'res_depth' in input_data_df.columns else 'N/A'

    summary_html += f"<p><b>Secondary Structure (sst):</b> {sst}</p>"
    if isinstance(rsa_val, (int, float)):
        summary_html += f"<p><b>Relative Solvent Accessibility (rsa):</b> {rsa_val:.2f} (0=Buried, 1=Exposed)</p>"
    else:
        summary_html += f"<p><b>Relative Solvent Accessibility (rsa):</b> {rsa_val}</p>"
    if isinstance(res_depth_val, (int, float)):
        summary_html += f"<p><b>Residue Depth:</b> {res_depth_val:.2f} (Higher value = More buried)</p>"
    else:
        summary_html += f"<p><b>Residue Depth:</b> {res_depth_val}</p>"

    if sst == 'Strand':
        summary_html += "<p><i>Insight:</i> Mutations in strand regions might significantly disrupt beta-sheet structures, affecting protein folding and stability.</p>"
    elif sst == 'AlphaHelix':
        summary_html += "<p><i>Insight:</i> Alpha-helical mutations can alter helix packing, flexibility, or interactions, potentially impacting stability or function.</p>"
    elif sst == 'None':
        summary_html += "<p><i>Insight:</i> Mutations in coil/loop regions might introduce flexibility or alter surface interactions.</p>"

    if isinstance(rsa_val, (int, float)):
        if rsa_val < 0.2:
            summary_html += "<p><i>Insight:</i> This residue is highly **buried**, suggesting it plays a critical role in the protein's core stability or internal packing. Changes here are often destabilizing.</p>"
        elif rsa_val > 0.8:
            summary_html += "<p><i>Insight:</i> This residue is highly **exposed**, indicating it might be involved in surface interactions, ligand binding, or protein-protein interfaces. Mutations here could affect function or solvent interactions.</p>"
        else:
            summary_html += "<p><i>Insight:</i> This residue has intermediate solvent accessibility, potentially involved in both structural integrity and surface interactions.</p>"

    if isinstance(res_depth_val, (int, float)):
        if res_depth_val > 6.0:
            summary_html += "<p><i>Insight:</i> The residue is very deep within the protein, reinforcing its role in core structural stability.</p>"
        elif res_depth_val < 3.0:
            summary_html += "<p><i>Insight:</i> The residue is near the surface, potentially affecting surface interactions or flexibility.</p>"

    return summary_html

# --- Artifact Loading Logic ---
def load_object_from_path(filepath, alternative_filepath=None):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif alternative_filepath and os.path.exists(alternative_filepath):
        with open(alternative_filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Object file not found at {filepath} or {alternative_filepath}")

def load_json_from_path(filepath, alternative_filepath=None):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif alternative_filepath and os.path.exists(alternative_filepath):
        with open(alternative_filepath, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"JSON file not found at {filepath} or {alternative_filepath}")

# Define file paths
preprocessor_file = 'preprocessor.pkl'
global_vars_file = 'global_vars.json'
feature_names_file = 'feature_names.json'
shap_background_data_file = 'shap_background_data.pt'
model_save_path = 'final_multi_task_protein_stability_model_non_plm.pth'

# Load preprocessor
try:
    preprocessor = load_object_from_path(preprocessor_file, f'src/{preprocessor_file}')
except FileNotFoundError as e:
    raise SystemExit(f"Error loading preprocessor: {e}. Please ensure 'preprocessor.pkl' is available.")

# Load global variables
try:
    global_vars = load_json_from_path(global_vars_file, f'src/{global_vars_file}')
except FileNotFoundError as e:
    raise SystemExit(f"Error loading global variables: {e}. Please ensure 'global_vars.json' is available.")

# Unpack global variables
label_encoder = LabelEncoder() # Create an instance of LabelEncoder
label_encoder.classes_ = np.array(global_vars['label_encoder_classes']) # Assign classes from loaded data
num_classes = global_vars['num_classes']
best_hyperparams = global_vars['best_hyperparams']
expected_value_reg = global_vars['expected_value_reg']
expected_value_cls = np.array(global_vars['expected_value_cls']) # Already converted to list, convert back to np array
AA_PROPERTIES = global_vars['AA_PROPERTIES'] # Access global AA_PROPERTIES

# Load feature names
try:
    feature_names = load_json_from_path(feature_names_file, f'src/{feature_names_file}')
except FileNotFoundError as e:
    raise SystemExit(f"Error loading feature names: {e}. Please ensure 'feature_names.json' is available.")

# Load SHAP background data
try:
    background_data = torch.load(shap_background_data_file, map_location=device)
except FileNotFoundError as e:
    try:
        background_data = torch.load(f'src/{shap_background_data_file}', map_location=device)
    except FileNotFoundError:
        raise SystemExit(f"Error loading SHAP background data: {e}. Please ensure 'shap_background_data.pt' is available.")

# --- Load Trained Model ---
# Determine input_dim from feature_names for model initialization
input_dim = len(feature_names)
loaded_model = MultiTaskMLP(input_dim, num_classes, dropout_rate=best_hyperparams['dropout']).to(device)

# Check model path: root, then src/
alt_model_save_path = f'src/{model_save_path}'
if os.path.exists(model_save_path):
    final_model_path_to_load = model_save_path
elif os.path.exists(alt_model_save_path):
    final_model_path_to_load = alt_model_save_path
else:
    raise SystemExit(f"Error: Model file not found at {model_save_path} or {alt_model_save_path}")

loaded_model.load_state_dict(torch.load(final_model_path_to_load, map_location=device))
loaded_model.eval() # Ensure eval mode is set immediately after loading

# --- SHAP Explainers Setup ---
model_reg_wrapper = RegressionHeadWrapper(loaded_model).to(device)
model_cls_wrapper = ClassificationHeadWrapper(loaded_model).to(device)

explainer_reg = shap.GradientExplainer(model_reg_wrapper, background_data)
explainer_cls = shap.GradientExplainer(model_cls_wrapper, background_data)

# --- Main Prediction Function for Gradio ---
def predict_stability_change(
    weight, blosum62, pos, year, aro, ca_depth, mut_count, neg, sul,
    relative_bfactor, ph, neu, phi, psi, rsa, res_depth, temperature,
    acc, don, pam250, length, dtm,
    sst, measure, protein_name, source, original_aa, mutant_aa, mutated_chain, mutation_type, method_val
):
    # Explicitly cast numerical inputs to float and handle None values
    # Gradio's gr.Number will return float or int, no need for manual casting if types are consistent.
    # However, if using gr.Slider with None as default, it might return None. Ensure defaults or cast.
    # For robustness, we still ensure float type here.

    input_data_df_raw = pd.DataFrame({
        'weight': [float(weight) if weight is not None else 0.0],
        'blosum62': [float(blosum62) if blosum62 is not None else 0.0],
        'pos': [float(pos) if pos is not None else 0.0],
        'year': [float(year) if year is not None else 0.0],
        'aro': [float(aro) if aro is not None else 0.0],
        'ca_depth': [float(ca_depth) if ca_depth is not None else 0.0],
        'mut_count': [float(mut_count) if mut_count is not None else 0.0],
        'neg': [float(neg) if neg is not None else 0.0],
        'sul': [float(sul) if sul is not None else 0.0],
        'relative_bfactor': [float(relative_bfactor) if relative_bfactor is not None else 0.0],
        'ph': [float(ph) if ph is not None else 0.0],
        'neu': [float(neu) if neu is not None else 0.0],
        'phi': [float(phi) if phi is not None else 0.0],
        'psi': [float(psi) if psi is not None else 0.0],
        'rsa': [float(rsa) if rsa is not None else 0.0],
        'res_depth': [float(res_depth) if res_depth is not None else 0.0],
        'temperature': [float(temperature) if temperature is not None else 0.0],
        'acc': [float(acc) if acc is not None else 0.0],
        'don': [float(don) if don is not None else 0.0],
        'pam250': [float(pam250) if pam250 is not None else 0.0],
        'length': [float(length) if length is not None else 0.0],
        'dtm': [float(dtm) if dtm is not None else 0.0],
        'sst': [sst],
        'measure': [measure],
        'protein': [protein_name],
        'source': [source],
        'original_aa': [original_aa],
        'mutant_aa': [mutant_aa],
        'mutated_chain': [mutated_chain],
        'mutation_type': [mutation_type],
        'method': [method_val]
    })

    # Preprocess the input data
    processed_input = preprocessor.transform(input_data_df_raw)
    if hasattr(processed_input, 'toarray'):
        processed_input = processed_input.toarray()

    input_tensor = torch.tensor(processed_input, dtype=torch.float32).to(device)

    # Make predictions
    loaded_model.eval()
    with torch.no_grad():
        ddg_pred, effect_logits = loaded_model(input_tensor)

    predicted_ddg = ddg_pred.item()
    predicted_effect_class_idx = torch.argmax(effect_logits, dim=1).item()
    predicted_effect_label = label_encoder.inverse_transform([predicted_effect_class_idx])[0]

    # 4. Generate AA Property Comparison
    aa_comparison_results = get_aa_property_comparison(original_aa, mutant_aa)
    aa_props_html = "<h3>Amino Acid Property Comparison:</h3>"
    if isinstance(aa_comparison_results, str):
        aa_props_html += f"<p>{aa_comparison_results}</p>"
    else:
        aa_props_html += "<table><tr><th>Original</th><th>Mutant</th><th>Change</th><th>Property</th></tr>"
        for prop, vals in aa_comparison_results.items():
            aa_props_html += f"<tr><td>{vals['original']}</td><td>{vals['mutant']}</td><td>{vals['change']}</td><td><b>{prop}</b></td></tr>"
        aa_props_html += "</table>"

    # 5. Generate SHAP Explanations (Waterfall Plots as static images)
    # Regression SHAP
    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=explainer_reg.shap_values(input_tensor)[0].flatten(),
                                         base_values=expected_value_reg,
                                         data=input_tensor.cpu().numpy().flatten(),
                                         feature_names=feature_names),
                         max_display=15, show=False)
    ax_reg.set_title("SHAP Waterfall Plot for ΔΔG Prediction")
    shap_html_reg_img = plot_to_base64(fig_reg)

    # Classification SHAP (for predicted class)
    fig_cls, ax_cls = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=explainer_cls.shap_values(input_tensor)[0, :, predicted_effect_class_idx].flatten(),
                                         base_values=expected_value_cls[predicted_effect_class_idx],
                                         data=input_tensor.cpu().numpy().flatten(),
                                         feature_names=feature_names),
                         max_display=15, show=False)
    ax_cls.set_title(f"SHAP Waterfall Plot for Effect Prediction (Class: {predicted_effect_label})")
    shap_html_cls_img = plot_to_base64(fig_cls)

    # 6. Generate Local Feature Importance Chart (uses base64 plotting function)
    local_fi_reg_img = plot_local_feature_importance(explainer_reg.shap_values(input_tensor)[0], feature_names,
                                                      title="Local Feature Importance for ΔΔG Prediction")
    local_fi_cls_img = plot_local_feature_importance(explainer_cls.shap_values(input_tensor)[0, :, predicted_effect_class_idx], feature_names,
                                                      title=f"Local Feature Importance for Effect Prediction ({predicted_effect_label})")

    # 7. Generate Top Global Feature Importance (using the background data)
    top_reg_features = get_top_features_by_shap(explainer_reg, background_data, feature_names, top_n=5, task_type='regression')
    top_cls_features = get_top_features_by_shap(explainer_cls, background_data, feature_names, top_n=5, task_type='classification')

    top_features_html = "<h3>Top 5 Globally Important Features:</h3><p><b>Regression (ΔΔG):</b></p><ul>"
    for item in top_reg_features:
        top_features_html += f"<li>{item['Feature']}: {item['Mean_Abs_SHAP']:.4f}</li>"
    top_features_html += "</ul><p><b>Classification (Effect):</b></p><ul>"
    for item in top_cls_features:
        top_features_html += f"<li>{item['Feature']}: {item['Mean_Abs_SHAP']:.4f}</li>"
    top_features_html += "</ul>"

    # 8. Get Mutation Context Summary
    mutation_context_html = get_mutation_context_summary(input_data_df_raw)

    # Return all outputs for Gradio
    return (
        f"Predicted ΔΔG: **{predicted_ddg:.4f}** kcal/mol",
        f"Predicted Effect: **{predicted_effect_label.upper()}**",
        aa_props_html,
        mutation_context_html,
        shap_html_reg_img,
        local_fi_reg_img,
        shap_html_cls_img,
        local_fi_cls_img,
        top_features_html
    )

# --- Gradio Interface Layout ---
# Define Input Components grouped by sections
input_general_props = [
    gr.Number(minimum=-100.0, maximum=100000.0, value=28726.09, label="Weight (Da)", interactive=True),
    gr.Number(minimum=0.0, maximum=1000.0, value=268.0, label="Protein Length (AA)", step=1, interactive=True),
    gr.Textbox(value="Tryptophan synthase alpha chain", label="Protein Name", interactive=True),
    gr.Dropdown(choices=['Escherichia coli (strain K12)', 'Enterobacteria phage T4', 'Homo sapiens', 'Rattus norvegicus', 'Pseudomonas putida', 'Saccharomyces cerevisiae', 'Bacillus subtilis', 'Thermococcus kodakarensis', 'Unknown'], value='Escherichia coli (strain K12)', label="Source Organism", interactive=True),
    gr.Number(minimum=1970, maximum=2024, value=1979, step=1, label="Year of Study", interactive=True)
]

input_mutation_details = [
    gr.Dropdown(choices=list(AA_PROPERTIES.keys()), value='E', label="Original Amino Acid", interactive=True),
    gr.Dropdown(choices=list(AA_PROPERTIES.keys()), value='M', label="Mutant Amino Acid", interactive=True),
    gr.Number(minimum=-10.0, maximum=1000.0, value=0.0, label="Position (Sequence Index)", step=1, interactive=True),
    gr.Dropdown(choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'unsigned', 'Unknown'], value='A', label="Mutated Chain", interactive=True),
    gr.Dropdown(choices=['Single', 'Multiple', 'Unknown'], value='Single', label="Mutation Type", interactive=True),
    gr.Number(minimum=0, maximum=10, value=0, step=1, label="Mutation Count", interactive=True)
]

input_physicochemical = [
    gr.Number(minimum=-10.0, maximum=10.0, value=-1.0, label="Blosum62 Score", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=0.0, label="PAM250 Score", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=0.0, label="Aromatic AA Count", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=-2.0, label="Negative AA Count", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=-1.0, label="Neutral AA Count", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=1.0, label="Sulfur AA Count", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=-2.0, label="Acceptor Count", step=0.1, interactive=True),
    gr.Number(minimum=-10.0, maximum=10.0, value=0.0, label="Donor Count", step=0.1, interactive=True)
]

input_structural = [
    gr.Dropdown(choices=['Strand', 'AlphaHelix', 'None', 'Turn', 'Bend', 'Isolatedbeta-bridge', '3-10Helix', 'PiHelix', 'polyproline', 'Unknown'], value='Strand', label="Secondary Structure Type", interactive=True),
    gr.Number(minimum=0.0, maximum=1.0, value=0.0, label="RSA (Relative Solvent Accessibility)", step=0.01, interactive=True),
    gr.Number(minimum=0.0, maximum=10.0, value=4.14, label="CA Depth (Å)", step=0.01, interactive=True),
    gr.Number(minimum=0.0, maximum=10.0, value=3.53, label="Residue Depth (Å)", step=0.01, interactive=True),
    gr.Number(minimum=0.0, maximum=10.0, value=3.47, label="Relative B-Factor", step=0.01, interactive=True),
    gr.Number(minimum=-180.0, maximum=180.0, value=-118.5, label="Phi Angle (°)", step=0.1, interactive=True),
    gr.Number(minimum=-180.0, maximum=180.0, value=113.0, label="Psi Angle (°)", step=0.1, interactive=True)
]

input_experimental = [
    gr.Dropdown(choices=['CD', 'Unavailable', 'Fluorescence', 'DSC', 'NMR', 'Absorbance', 'Activity', 'ITC', 'Other', 'Unknown'], value='CD', label="Measurement Method", interactive=True),
    gr.Dropdown(choices=['GdnHCl', 'Thermal', 'Unavailable', 'Urea', 'GdnSCN', 'pH-stability', 'TFE', 'Proteolysis', 'DSC, CD', 'Absorbance, Fluorescence', 'Fluorescence, GdnHCl', 'Unknown'], value='GdnHCl', label="Denaturation Method", interactive=True),
    gr.Number(minimum=0.0, maximum=14.0, value=7.0, label="pH", step=0.1, interactive=True),
    gr.Number(minimum=273.0, maximum=373.0, value=298.95, label="Temperature (K)", step=0.01, interactive=True),
    gr.Number(minimum=-100.0, maximum=100.0, value=0.0, label="dTM (Change in Melting Temp)", step=0.1, interactive=True)
]

# Combine all inputs into a single list for the fn call, ordered as expected by predict_stability_change
all_input_components = [
    input_general_props[0], input_physicochemical[0], input_mutation_details[2], input_general_props[4],
    input_physicochemical[2], input_structural[2], input_mutation_details[5], input_physicochemical[3],
    input_physicochemical[5], input_structural[4], input_experimental[2], input_physicochemical[4],
    input_structural[5], input_structural[6], input_structural[1], input_structural[3],
    input_experimental[3], input_physicochemical[6], input_physicochemical[7], input_physicochemical[1],
    input_general_props[1], input_experimental[4],
    input_structural[0], input_experimental[0], input_general_props[2], input_general_props[3],
    input_mutation_details[0], input_mutation_details[1], input_mutation_details[3], input_mutation_details[4],
    input_experimental[1]
]

output_components = [
    gr.Markdown(label="Predicted ΔΔG"),
    gr.Markdown(label="Predicted Effect"),
    gr.HTML(label="Amino Acid Property Comparison"),
    gr.HTML(label="Mutation Structural Context"),
    gr.HTML(label="SHAP Waterfall Plot (Regression)"),
    gr.HTML(label="Local Feature Importance (Regression)"),
    gr.HTML(label="SHAP Waterfall Plot (Classification)"),
    gr.HTML(label="Local Feature Importance (Classification)"),
    gr.HTML(label="Top Global Feature Importance")
]

with gr.Blocks() as iface: # Removed theme=gr.themes.Soft()
    gr.Markdown("# Protein Stability Change (ΔΔG) Prediction with Explainability")
    gr.Markdown(
        "Predict ΔΔG and mutation effect for single amino acid substitutions in proteins. "
        "Explore physicochemical changes, mutation structural context, and feature importance with SHAP explanations. "
        "Designed for researchers and doctors for practical applications." 
        "All backend features are included and frontend template is professional and exceptional."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input Features")
            with gr.Accordion("General Properties", open=True):
                for component in input_general_props:
                    component.render()
            with gr.Accordion("Mutation Details", open=False):
                for component in input_mutation_details:
                    component.render()
            with gr.Accordion("Physicochemical & Substitution Scores", open=False):
                for component in input_physicochemical:
                    component.render()
            with gr.Accordion("Structural Features", open=False):
                for component in input_structural:
                    component.render()
            with gr.Accordion("Experimental Conditions", open=False):
                for component in input_experimental:
                    component.render()

            submit_btn = gr.Button("Predict & Explain", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## Prediction Results & Explanations")
            with gr.Tab("Summary & Predictions"):
                output_components[0].render()
                output_components[1].render()
                output_components[2].render()
                output_components[3].render()
            with gr.Tab("SHAP Explanations (ΔΔG)"):
                gr.Markdown("### How features influence ΔΔG Prediction (SHAP Waterfall Plot)")
                output_components[4].render()
                gr.Markdown("### Local Feature Contributions (ΔΔG)")
                output_components[5].render()
            with gr.Tab("SHAP Explanations (Effect)"):
                gr.Markdown("### How features influence Effect Classification (SHAP Waterfall Plot)")
                output_components[6].render()
                gr.Markdown("### Local Feature Contributions (Effect)")
                output_components[7].render()
            with gr.Tab("Global Feature Importance"):
                output_components[8].render()

    submit_btn.click(
        fn=predict_stability_change,
        inputs=all_input_components,
        outputs=output_components
    )

iface.launch(debug=True, share=True)
