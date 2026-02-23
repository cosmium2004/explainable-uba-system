import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
import re
import argparse
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

def load_results(file_path=None):
    """Load prediction results from a JSON file."""
    if file_path is None:
        # Find the most recent results file
        results_dir = "d:/MiniProject/LLM/results"
        json_files = [f for f in os.listdir(results_dir) if f.startswith("prediction_results_") and f.endswith(".json")]
        
        if not json_files:
            print("No prediction results found. Please run uba_demo.py first.")
            return None
        
        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        file_path = os.path.join(results_dir, json_files[0])
        print(f"Using most recent results file: {file_path}")
    
    # Load the JSON file
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} prediction results.")
        return results
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def create_output_dir():
    """Create output directory for visualizations if it doesn't exist."""
    output_dir = "d:/MiniProject/LLM/results/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def confidence_distribution(results, output_dir):
    """Create histogram of confidence score distribution."""
    # Extract confidence scores
    anomaly_scores = [r['llm_confidence'] for r in results if r['llm_prediction'] == 'Anomaly']
    normal_scores = [r['llm_confidence'] for r in results if r['llm_prediction'] == 'Normal']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores, alpha=0.7, label='Anomaly', color='red', bins=10, range=(0, 1))
    plt.hist(normal_scores, alpha=0.7, label='Normal', color='green', bins=10, range=(0, 1))
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    output_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confidence distribution to {output_path}")
    
    return output_path

def risk_factors_heatmap(results, output_dir):
    """Create heatmap of risk factors."""
    # Define risk factors to look for
    risk_factors = ['timing', 'access pattern', 'sensitive data', 'data volume', 'authentication']
    
    # Create a matrix of risk factors
    matrix = []
    texts = []
    
    for result in results:
        text = result['text']
        texts.append(text[:30] + '...' if len(text) > 30 else text)
        
        explanation = result.get('llm_explanation', '')
        row = []
        
        for factor in risk_factors:
            if factor.lower() in explanation.lower():
                row.append(1)
            else:
                row.append(0)
        
        matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(matrix, columns=risk_factors, index=texts)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, cmap='YlOrRd', cbar_kws={'label': 'Present in Explanation'})
    plt.title('Risk Factors Contributing to Predictions')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'risk_factors_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved risk factors heatmap to {output_path}")
    
    return output_path

def risk_vs_confidence(results, output_dir):
    """Create scatter plot of confidence vs. risk factors."""
    # Count risk factors in each explanation
    risk_factors = []
    confidence_scores = []
    is_anomaly = []
    
    for result in results:
        explanation = result.get('llm_explanation', '')
        # Count risk factors by looking for commas in the explanation
        if "due to " in explanation and "," in explanation.split("due to ")[1]:
            count = explanation.split("due to ")[1].count(",") + 1
        elif "due to " in explanation:
            count = 1
        else:
            count = 0
        
        risk_factors.append(count)
        confidence_scores.append(result['llm_confidence'])
        is_anomaly.append(1 if result['llm_prediction'] == 'Anomaly' else 0)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter([risk_factors[i] for i in range(len(risk_factors)) if is_anomaly[i] == 1], 
                [confidence_scores[i] for i in range(len(confidence_scores)) if is_anomaly[i] == 1], 
                color='red', label='Anomaly', alpha=0.7, s=100)
    plt.scatter([risk_factors[i] for i in range(len(risk_factors)) if is_anomaly[i] == 0], 
                [confidence_scores[i] for i in range(len(confidence_scores)) if is_anomaly[i] == 0], 
                color='green', label='Normal', alpha=0.7, s=100)
    
    # Add text labels for each point
    for i, txt in enumerate([r['text'][:20] + '...' for r in results]):
        plt.annotate(txt, (risk_factors[i], confidence_scores[i]), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Number of Risk Factors')
    plt.ylabel('Confidence Score')
    plt.title('Relationship Between Risk Factors and Confidence')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    output_path = os.path.join(output_dir, 'risk_vs_confidence.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved risk vs confidence plot to {output_path}")
    
    return output_path

def model_comparison_radar(results, output_dir):
    """Create radar chart for multi-model comparison."""
    # Check if we have both LLM and RF predictions
    has_rf = all('rf_confidence' in r for r in results)
    
    if not has_rf:
        print("Skipping radar chart - RF predictions not available")
        return None
    
    # Select a subset of results for clarity (max 8)
    sample_results = results[:min(8, len(results))]
    
    # Create radar chart
    categories = [f"Sample {i+1}" for i in range(len(sample_results))]
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Extract confidence scores
    llm_values = [r["llm_confidence"] for r in sample_results]
    llm_values += llm_values[:1]  # Close the loop
    rf_values = [r["rf_confidence"] for r in sample_results]
    rf_values += rf_values[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw the LLM line
    ax.plot(angles, llm_values, linewidth=2, linestyle='solid', label='LLM Model')
    ax.fill(angles, llm_values, alpha=0.25)
    
    # Draw the RF line
    ax.plot(angles, rf_values, linewidth=2, linestyle='solid', label='RF Model')
    ax.fill(angles, rf_values, alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Confidence Comparison', size=15)
    
    # Save figure
    output_path = os.path.join(output_dir, 'model_comparison_radar.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model comparison radar to {output_path}")
    
    return output_path

def interactive_dashboard(results, output_dir):
    """Create interactive dashboard with Plotly."""
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create interactive bar chart
    fig = px.bar(df, 
                x=[r['text'][:20] + '...' for r in results],  # Truncate text for display
                y='llm_confidence', 
                color='llm_prediction',
                hover_data=['text', 'llm_explanation'],
                title='Anomaly Detection Confidence Scores',
                labels={'llm_confidence': 'Confidence Score', 'x': 'Event Description'})
    
    fig.update_layout(xaxis_tickangle=-45)
    
    # Save figure
    output_path = os.path.join(output_dir, 'interactive_dashboard.html')
    fig.write_html(output_path)
    print(f"Saved interactive dashboard to {output_path}")
    
    return output_path

def time_series_visualization(results, output_dir):
    """Create time-series visualization of predictions."""
    # Create synthetic timestamps (since data doesn't have real timestamps)
    base_time = datetime.now()
    timestamps = [base_time - timedelta(hours=i*2) for i in range(len(results))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'confidence': [r['llm_confidence'] for r in results],
        'prediction': [r['llm_prediction'] for r in results],
        'text': [r['text'] for r in results]
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['confidence'], 'o-', color='blue', alpha=0.7)
    
    # Color points based on prediction
    for i, row in df.iterrows():
        if row['prediction'] == 'Anomaly':
            plt.plot(row['timestamp'], row['confidence'], 'o', color='red', markersize=10)
        else:
            plt.plot(row['timestamp'], row['confidence'], 'o', color='green', markersize=10)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.title('Anomaly Detection Over Time')
    plt.xlabel('Time')
    plt.ylabel('Confidence Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add threshold line
    plt.axhline(y=0.5, color='orange', linestyle='--', label='Threshold (0.5)')
    
    # Add legend
    plt.legend(['Confidence Trend', 'Anomaly', 'Normal', 'Threshold'], loc='best')
    
    # Save figure
    output_path = os.path.join(output_dir, 'time_series_visualization.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved time series visualization to {output_path}")
    
    return output_path

def explanation_wordcloud(results, output_dir):
    """Create word cloud from explanations."""
    try:
        from wordcloud import WordCloud
        
        # Combine all explanations
        all_explanations = " ".join([r.get('llm_explanation', '') for r in results])
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(all_explanations)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Terms in Explanations')
        
        # Save figure
        output_path = os.path.join(output_dir, 'explanation_wordcloud.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved explanation wordcloud to {output_path}")
        
        return output_path
    except ImportError:
        print("WordCloud package not installed. Skipping wordcloud visualization.")
        print("Install with: pip install wordcloud")
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize UBA prediction results')
    parser.add_argument('--file', type=str, help='Path to prediction results JSON file')
    parser.add_argument('--all', action='store_true', help='Generate all visualizations')
    parser.add_argument('--dist', action='store_true', help='Generate confidence distribution')
    parser.add_argument('--heatmap', action='store_true', help='Generate risk factors heatmap')
    parser.add_argument('--scatter', action='store_true', help='Generate risk vs confidence scatter plot')
    parser.add_argument('--radar', action='store_true', help='Generate model comparison radar chart')
    parser.add_argument('--dashboard', action='store_true', help='Generate interactive dashboard')
    parser.add_argument('--timeseries', action='store_true', help='Generate time series visualization')
    parser.add_argument('--wordcloud', action='store_true', help='Generate explanation wordcloud')
    return parser.parse_args()

def create_comprehensive_dashboard(results, output_dir, metadata=None):
    """Create a comprehensive interactive dashboard with educational elements.
    
    Args:
        results: List of prediction results
        output_dir: Directory to save the dashboard
        metadata: Optional metadata about the input type and scenario
    """
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Get input type from metadata
    input_type = metadata.get('input_type', 'unknown') if metadata else 'unknown'
    scenario_type = metadata.get('scenario_type', None) if metadata else None
    
    # Create a Plotly figure with multiple subplots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Determine subplot configuration based on input type
    if input_type == 'text':
        # For single text input, focus on explanation and risk factors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Anomaly Detection Result", 
                "Risk Factors Identified",
                "Confidence Comparison", 
                "Explanation Analysis"
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.25,  # Increased from 0.15
            horizontal_spacing=0.15  # Increased from 0.1
        )
    elif input_type == 'scenario':
        # For scenario input, focus on patterns and comparisons
        scenario_title = "Threat Scenario Analysis"
        if scenario_type:
            scenario_title = f"{scenario_type.title()} {scenario_title}"
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                scenario_title, 
                "Normal vs. Anomalous Events",
                "Risk Factors Distribution", 
                "Model Confidence Comparison"
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.25,  # Increased from 0.15
            horizontal_spacing=0.15  # Increased from 0.1
        )
    else:  # sample or unknown
        # For sample texts, use the original comprehensive view
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Anomaly Detection Confidence", 
                "Risk Factors vs. Confidence",
                "Anomaly Detection Over Time", 
                "Explanation Word Frequency"
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.25,  # Increased from 0.15
            horizontal_spacing=0.15  # Increased from 0.1
        )

    # Reduce font size for subplot titles to prevent overlap
    fig.update_annotations(font_size=12)
    
    # Generate colors based on predictions
    colors = ['red' if pred == 'Anomaly' else 'green' for pred in df['llm_prediction']]
    
    # Create hover texts
    hover_texts = []
    for i, r in enumerate(results):
        hover_text = f"<b>Event:</b> {r['text']}<br>"
        hover_text += f"<b>Prediction:</b> {r['llm_prediction']}<br>"
        hover_text += f"<b>Confidence:</b> {r['llm_confidence']:.2f}<br>"
        hover_text += f"<b>Explanation:</b> {r['llm_explanation']}"
        hover_texts.append(hover_text)
    
    # Create visualizations based on input type
    if input_type == 'text':
        # Single text input visualizations
        
        # 1. Gauge chart for the prediction
        # For gauge charts (single text input)
        fig.add_trace(
            go.Indicator(
                value=df['llm_confidence'].iloc[0],
                title={'text': f"Prediction: {df['llm_prediction'].iloc[0]}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': colors[0]},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Risk factors bar chart
        # Extract risk factors from explanation
        risk_factors = []
        risk_levels = []
        
        explanation = df['llm_explanation'].iloc[0]
        
        # More robust parsing of risk factors
        if "due to " in explanation:
            factors_text = explanation.split("due to ")[1]
            # Remove trailing period if present
            if factors_text.endswith("."):
                factors_text = factors_text[:-1]
                
            # Extract factors separated by commas
            if "," in factors_text:
                factors = [f.strip() for f in factors_text.split(",")]
                for factor in factors:
                    if ":" in factor:
                        factor_name, level = factor.split(":", 1)
                        risk_factors.append(factor_name.strip())
                        if "critical" in level.lower():
                            risk_levels.append(1.0)
                        elif "high" in level.lower():
                            risk_levels.append(0.8)
                        elif "medium" in level.lower():
                            risk_levels.append(0.6)
                        elif "low" in level.lower():
                            risk_levels.append(0.4)
                        else:
                            risk_levels.append(0.5)
                    else:
                        risk_factors.append(factor)
                        risk_levels.append(0.7)  # Default to medium-high if no level specified
            else:
                risk_factors.append(factors_text)
                risk_levels.append(0.7)  # Default to medium-high if no level specified
        elif "flagged as anomalous" in explanation.lower():
            # Try to extract factors from other explanation formats
            parts = explanation.lower().split("flagged as anomalous")[1]
            if "based on" in parts:
                factor = parts.split("based on")[1].strip()
                if factor.endswith("."):
                    factor = factor[:-1]
                risk_factors.append(factor)
                risk_levels.append(0.7)
        
        # If no risk factors found, create placeholder
        if not risk_factors:
            risk_factors = ["No specific risk factors identified"]
            risk_levels = [0.1]
        
        # Create bar chart of risk factors
        fig.add_trace(
            go.Bar(
                x=risk_factors,
                y=risk_levels,
                marker_color=['rgba(255,0,0,' + str(level) + ')' for level in risk_levels],
                text=[f"{level*100:.0f}%" for level in risk_levels],
                textposition="auto",
                name="Risk Factors"
            ),
            row=1, col=2
        )
        
        # 3. Model comparison if RF model is available
        if 'rf_prediction' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=["LLM", "Random Forest", "Hybrid"],
                    y=[df['llm_confidence'].iloc[0], df['rf_confidence'].iloc[0], df['hybrid_score'].iloc[0]],
                    marker_color=['blue', 'orange', 'purple'],
                    text=[f"{df['llm_prediction'].iloc[0]}", f"{df['rf_prediction'].iloc[0]}", f"{df['hybrid_prediction'].iloc[0]}"],
                    textposition="auto",
                    name="Model Comparison"
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=["LLM"],
                    y=[df['llm_confidence'].iloc[0]],
                    marker_color=['blue'],
                    text=[f"{df['llm_prediction'].iloc[0]}"],
                    textposition="auto",
                    name="Model Prediction"
                ),
                row=2, col=1
            )
        
        # 4. Explanation analysis - key terms
        from collections import Counter
        import re
        
        # Extract words from explanation
        explanation = df['llm_explanation'].iloc[0]
        words = re.findall(r'\b\w+\b', explanation.lower())
        # Remove common stopwords
        stopwords = ['the', 'and', 'to', 'of', 'with', 'in', 'as', 'due', 'is', 'a', 'for', 'based', 'some', 'overall']
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        word_counts = Counter(filtered_words).most_common(10)
        
        if word_counts:
            fig.add_trace(
                go.Bar(
                    x=[word for word, count in word_counts],
                    y=[count for word, count in word_counts],
                    marker_color='rgba(66, 135, 245, 0.8)',
                    name="Key Terms"
                ),
                row=2, col=2
            )
        else:
            fig.add_annotation(
                x=0.5, y=0.5,
                xref="x4", yref="y4",
                text="No significant terms found in explanation",
                showarrow=False,
                font=dict(size=12),
                row=2, col=2
            )
    
    elif input_type == 'scenario':
        # Scenario-specific visualizations
        
        # 1. Confidence scores by sample
        fig.add_trace(
            go.Bar(
                x=[i for i in range(len(results))],
                y=df['llm_confidence'],
                marker_color=colors,
                text=df['llm_prediction'],
                hovertext=hover_texts,
                hoverinfo='text',
                name='Confidence Scores'
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0.5,
            x1=len(results)-0.5,
            y1=0.5,
            line=dict(color="orange", width=2, dash="dash"),
            row=1, col=1
        )
        
        # 2. Pie chart of normal vs anomalous events
        normal_count = sum(1 for pred in df['llm_prediction'] if pred == 'Normal')
        anomaly_count = len(results) - normal_count
        
        fig.add_trace(
            go.Pie(
                labels=["Normal", "Anomaly"],
                values=[normal_count, anomaly_count],
                marker=dict(colors=['green', 'red']),
                textinfo="label+percent",
                hole=0.3,
                name="Event Distribution"
            ),
            row=1, col=2
        )
        
        # 3. Risk factors heatmap
        # Define risk factors to look for
        risk_factors = ['timing', 'access pattern', 'sensitive data', 'data volume', 'authentication', 'location', 'frequency']
        
        # Create a matrix of risk factors
        matrix = []
        
        for result in results:
            explanation = result.get('llm_explanation', '')
            row = []
            
            for factor in risk_factors:
                if factor.lower() in explanation.lower():
                    row.append(1)
                else:
                    row.append(0)
            
            matrix.append(row)
        
        # Create heatmap
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=risk_factors,
                y=[i for i in range(len(results))],
                colorscale='YlOrRd',
                showscale=True,
                name="Risk Factors"
            ),
            row=2, col=1
        )
        
        # 4. Model confidence comparison scatter plot
        if 'rf_confidence' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['llm_confidence'],
                    y=df['rf_confidence'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=colors,
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=hover_texts,
                    hoverinfo='text',
                    name='LLM vs RF'
                ),
                row=2, col=2
            )
            
            # Add diagonal line (perfect agreement)
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="gray", width=2, dash="dash"),
                row=2, col=2
            )
            
            # Add threshold lines
            fig.add_shape(
                type="line",
                x0=0.5,
                y0=0,
                x1=0.5,
                y1=1,
                line=dict(color="blue", width=1, dash="dot"),
                row=2, col=2
            )
            fig.add_shape(
                type="line",
                x0=0,
                y0=0.5,
                x1=1,
                y1=0.5,
                line=dict(color="orange", width=1, dash="dot"),
                row=2, col=2
            )
        else:
            # If RF model not available, show time series instead
            base_time = datetime.now()
            timestamps = [base_time - timedelta(hours=i*2) for i in range(len(results))]
            timestamps.reverse()  # Reverse to show chronological order
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=df['llm_confidence'],
                    mode='lines+markers',
                    marker=dict(color=colors, size=10),
                    line=dict(color='royalblue'),
                    text=hover_texts,
                    hoverinfo='text',
                    name='Confidence Over Time'
                ),
                row=2, col=2
            )
    
    else:  # sample or unknown - use original comprehensive dashboard
        # 1. Confidence Scores Bar Chart
        fig.add_trace(
            go.Bar(
                x=[i for i in range(len(results))],
                y=df['llm_confidence'],
                marker_color=colors,
                text=df['llm_prediction'],
                hovertext=hover_texts,
                hoverinfo='text',
                name='LLM Confidence'
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=0.5,
            x1=len(results)-0.5,
            y1=0.5,
            line=dict(color="orange", width=2, dash="dash"),
            row=1, col=1
        )
        
        # 2. Risk vs Confidence Scatter Plot
        risk_factors = []
        for result in results:
            explanation = result.get('llm_explanation', '')
            if "due to " in explanation and "," in explanation.split("due to ")[1]:
                count = explanation.split("due to ")[1].count(",") + 1
            elif "due to " in explanation:
                count = 1
            else:
                count = 0
            risk_factors.append(count)
        
        fig.add_trace(
            go.Scatter(
                x=risk_factors,
                y=df['llm_confidence'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=hover_texts,
                hoverinfo='text',
                name='Risk vs Confidence'
            ),
            row=1, col=2
        )
        
        # 3. Time Series Visualization
        base_time = datetime.now()
        timestamps = [base_time - timedelta(hours=i*2) for i in range(len(results))]
        timestamps.reverse()  # Reverse to show chronological order
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=df['llm_confidence'],
                mode='lines+markers',
                marker=dict(color=colors, size=10),
                line=dict(color='royalblue'),
                text=hover_texts,
                hoverinfo='text',
                name='Confidence Over Time'
            ),
            row=2, col=1
        )
        
        # 4. Explanation Word Frequency
        from collections import Counter
        import re
        
        # Extract words from explanations
        all_explanations = " ".join([r.get('llm_explanation', '') for r in results])
        words = re.findall(r'\b\w+\b', all_explanations.lower())
        # Remove common stopwords
        stopwords = ['the', 'and', 'to', 'of', 'with', 'in', 'as', 'due', 'is', 'a', 'for', 'based', 'some', 'overall']
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        word_counts = Counter(filtered_words).most_common(10)
        
        # Use a color gradient based on frequency
        max_count = word_counts[0][1] if word_counts else 1
        word_colors = [f'rgba(66, 135, 245, {count/max_count})' for _, count in word_counts]
        
        fig.add_trace(
            go.Bar(
                x=[word for word, count in word_counts],
                y=[count for word, count in word_counts],
                marker_color=word_colors,
                name='Word Frequency'
            ),
            row=2, col=2
        )
    
    # Add title based on input type
    if input_type == 'text':
        title = "Single Event Analysis Dashboard"
    elif input_type == 'scenario':
        title = f"{scenario_type.title()} Threat Scenario Analysis Dashboard"
    else:  # sample
        title = "Sample Events Analysis Dashboard"
    
    # Add educational annotations
    fig.add_annotation(
        x=0.5, y=1.12,
        xref="paper", yref="paper",
        text=f"<b>{title}</b><br>This dashboard demonstrates how AI models can detect suspicious user behaviors",
        showarrow=False,
        font=dict(size=14),
        align="center"
    )
    
    # Add timestamp and metadata
    timestamp_str = metadata.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")) if metadata else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text=f"<i>Analysis generated at {timestamp_str}</i>",
        showarrow=False,
        font=dict(size=10),
        align="center"
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=False,
        template="plotly_white",
        margin=dict(t=100, b=100)
    )
    
    # Update axes labels based on input type
    if input_type == 'text':
        if 'rf_prediction' in df.columns:
            fig.update_xaxes(title_text="Model", row=2, col=1)
            fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Model", row=2, col=1)
            fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Risk Factor", row=1, col=2)
        fig.update_yaxes(title_text="Risk Level", row=1, col=2)
        
        fig.update_xaxes(title_text="Term", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    elif input_type == 'scenario':
        fig.update_xaxes(title_text="Sample ID", row=1, col=1)
        fig.update_yaxes(title_text="Confidence Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Risk Factor", row=2, col=1)
        fig.update_yaxes(title_text="Sample ID", row=2, col=1)
        
        if 'rf_confidence' in df.columns:
            fig.update_xaxes(title_text="LLM Confidence", row=2, col=2)
            fig.update_yaxes(title_text="RF Confidence", row=2, col=2)
        else:
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_yaxes(title_text="Confidence Score", row=2, col=2)
    
    else:  # sample
        fig.update_xaxes(title_text="Sample ID", row=1, col=1)
        fig.update_yaxes(title_text="Confidence Score", row=1, col=1)
        
        fig.update_xaxes(title_text="Number of Risk Factors", row=1, col=2)
        fig.update_yaxes(title_text="Confidence Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
        
        fig.update_xaxes(title_text="Word", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Save to HTML file
    output_path = os.path.join(output_dir, 'comprehensive_dashboard.html')
    fig.write_html(output_path)
    print(f"Saved comprehensive dashboard to {output_path}")
    
    return output_path

def main():
    """Main function."""
    args = parse_args()
    
    # Load results
    results = load_results(args.file)
    if results is None:
        return
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Determine which visualizations to generate
    generate_all = args.all or not any([args.dist, args.heatmap, args.scatter, 
                                       args.radar, args.dashboard, args.timeseries, 
                                       args.wordcloud])
    
    # Generate visualizations
    outputs = {}
    
    if generate_all or args.dist:
        outputs['confidence_distribution'] = confidence_distribution(results, output_dir)
    
    if generate_all or args.heatmap:
        outputs['risk_factors_heatmap'] = risk_factors_heatmap(results, output_dir)
    
    if generate_all or args.scatter:
        outputs['risk_vs_confidence'] = risk_vs_confidence(results, output_dir)
    
    if generate_all or args.radar:
        outputs['model_comparison_radar'] = model_comparison_radar(results, output_dir)
    
    if generate_all or args.dashboard:
        outputs['interactive_dashboard'] = interactive_dashboard(results, output_dir)
        
    # Always generate the comprehensive dashboard
    outputs['comprehensive_dashboard'] = create_comprehensive_dashboard(results, output_dir)
    
    if generate_all or args.timeseries:
        outputs['time_series_visualization'] = time_series_visualization(results, output_dir)
    
    if generate_all or args.wordcloud:
        outputs['explanation_wordcloud'] = explanation_wordcloud(results, output_dir)
    
    # Print summary
    print("\nVisualization Summary:")
    for name, path in outputs.items():
        if path:
            print(f"- {name}: {path}")
    
    print(f"\nAll visualizations saved to {output_dir}")
    
    return outputs

if __name__ == "__main__":
    main()