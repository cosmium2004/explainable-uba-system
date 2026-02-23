import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory

# Import functions from existing scripts
from uba_demo import load_models, predict_with_llm, predict_with_rf, display_results, get_sample_texts, get_scenario_texts
from visualize_results import create_comprehensive_dashboard, create_output_dir

app = Flask(__name__)

# Initialize models at startup
print("Loading models...")
llm_classifier, rf_model = load_models()

# Set paths
results_path = "d:/MiniProject/LLM/results"

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get input from form
    input_type = request.form.get('input_type', 'text')
    
    if input_type == 'text':
        text = request.form.get('text', '')
        if text:
            texts = [text]
        else:
            return jsonify({'error': 'No text provided'})
    elif input_type == 'scenario':
        scenario = request.form.get('scenario', 'mixed')
        texts = get_scenario_texts(scenario)
        # Store scenario type for visualization
        scenario_type = scenario
    else:  # sample
        texts = get_sample_texts()
    
    # Make predictions
    llm_predictions = predict_with_llm(llm_classifier, texts)
    
    if rf_model:
        rf_predictions = predict_with_rf(rf_model, texts)
    else:
        rf_predictions = None
    
    # Process results
    results = []
    for i, (text, llm_pred) in enumerate(zip(texts, llm_predictions)):
        is_anomaly_llm = 1 if llm_pred['score'] >= 0.5 and llm_pred['label'] == 'LABEL_1' else 0
        confidence_llm = llm_pred['score']
        
        result = {
            'text': text,
            'llm_prediction': 'Anomaly' if is_anomaly_llm else 'Normal',
            'llm_confidence': confidence_llm,
            'llm_explanation': llm_pred.get('explanation', 'No explanation available')
        }
        
        # Add RF predictions if available
        if rf_predictions:
            rf_confidence = rf_predictions[i]['score']
            rf_explanation = rf_predictions[i].get('explanation', 'No explanation available')
            is_anomaly_rf = 1 if rf_confidence >= 0.5 else 0
            result['rf_prediction'] = 'Anomaly' if is_anomaly_rf else 'Normal'
            result['rf_confidence'] = rf_confidence
            result['rf_explanation'] = rf_explanation
            
            # Add hybrid prediction
            llm_weight = confidence_llm if is_anomaly_llm else (1 - confidence_llm)
            rf_weight = rf_confidence if is_anomaly_rf else (1 - rf_confidence)
            hybrid_score = (llm_weight + rf_weight) / 2
            is_hybrid_anomaly = hybrid_score >= 0.5
            
            result['hybrid_prediction'] = 'Anomaly' if is_hybrid_anomaly else 'Normal'
            result['hybrid_score'] = hybrid_score
        
        results.append(result)
    
    # Create metadata for visualization customization
    metadata = {
        'input_type': input_type,
        'scenario_type': scenario_type if input_type == 'scenario' else None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'sample_count': len(texts)
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_results_{timestamp}.json"
    filepath = os.path.join(results_path, filename)
    
    # Save results with metadata
    with open(filepath, 'w') as f:
        json.dump({'results': results, 'metadata': metadata}, f, indent=4)
    
    # Generate dashboard - pass metadata directly to the function
    output_dir = create_output_dir()
    dashboard_path = create_comprehensive_dashboard(results, output_dir, metadata)
    
    # Return path to dashboard - modified to use results/visualizations instead of static/visualizations
    dashboard_url = f"/results/visualizations/{os.path.basename(dashboard_path)}"
    return jsonify({'dashboard_url': dashboard_url, 'results': results})

@app.route('/results/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory('d:/MiniProject/LLM/results/visualizations', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/visualizations', exist_ok=True)
    
    # Copy existing visualizations to static folder
    import shutil
    vis_dir = "d:/MiniProject/LLM/results/visualizations"
    if os.path.exists(vis_dir):
        for file in os.listdir(vis_dir):
            if file.endswith('.html'):
                shutil.copy(os.path.join(vis_dir, file), 'static/visualizations/')
    
    app.run(debug=True, port=5000)