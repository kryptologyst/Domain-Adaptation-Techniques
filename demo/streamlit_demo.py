"""Interactive demo for domain adaptation using Streamlit."""

import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from domain_adaptation.data import load_mnist_data, load_usps_data
from domain_adaptation.models import (
    AdversarialDomainAdaptationModel,
    CORALModel,
    SourceOnlyModel,
    TargetOnlyModel,
)
from domain_adaptation.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Domain Adaptation Techniques Demo",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety Disclaimer</h4>
    <p><strong>This is a research/educational demonstration only.</strong></p>
    <ul>
        <li>Not intended for production use or real-world decision making</li>
        <li>Models are trained on limited datasets and may not generalize</li>
        <li>Results should be interpreted with caution</li>
        <li>Human oversight is required for any critical applications</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔄 Domain Adaptation Techniques Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Source Only", "Target Only", "Adversarial", "CORAL"],
    help="Choose the domain adaptation method to explore"
)

# Parameters
st.sidebar.subheader("Model Parameters")
feature_dim = st.sidebar.slider("Feature Dimension", 64, 256, 128)
dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.8, 0.5)
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)

# Training parameters
st.sidebar.subheader("Training Parameters")
num_epochs = st.sidebar.slider("Number of Epochs", 10, 100, 20)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 64)

# Loss weights (for adversarial model)
if model_type == "Adversarial":
    st.sidebar.subheader("Loss Weights")
    adv_weight = st.sidebar.slider("Adversarial Weight", 0.1, 2.0, 1.0)
elif model_type == "CORAL":
    st.sidebar.subheader("Loss Weights")
    coral_weight = st.sidebar.slider("CORAL Weight", 0.01, 1.0, 0.1)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Model Comparison", "🔍 Analysis", "📈 Results"])

with tab1:
    st.header("Domain Adaptation Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What is Domain Adaptation?")
        st.markdown("""
        Domain adaptation is a machine learning technique that enables models trained on one domain 
        (source) to perform well on a different but related domain (target), even when labeled 
        data is scarce in the target domain.
        
        **Key Challenges:**
        - Domain shift between source and target
        - Limited labeled data in target domain
        - Maintaining performance on both domains
        """)
        
        st.subheader("Methods Implemented")
        methods = {
            "Source Only": "Baseline trained only on source domain",
            "Target Only": "Baseline trained only on target domain", 
            "Adversarial": "Uses adversarial training to learn domain-invariant features",
            "CORAL": "Aligns feature distributions using correlation alignment"
        }
        
        for method, description in methods.items():
            st.markdown(f"**{method}:** {description}")
    
    with col2:
        st.subheader("Dataset Information")
        st.markdown("""
        **Source Domain:** MNIST Handwritten Digits
        - 60,000 training images
        - 10 classes (0-9)
        - 28x28 grayscale images
        
        **Target Domain:** USPS Handwritten Digits  
        - 7,291 training images
        - 10 classes (0-9)
        - 16x16 grayscale images
        
        **Domain Shift:** Different image sizes, writing styles, and digit appearances
        """)
        
        # Show sample images
        st.subheader("Sample Images")
        try:
            # Load sample data
            set_seed(42)
            device = get_device()
            
            # Load a few samples
            source_data, source_targets = load_mnist_data(train=True)
            target_data, target_targets = load_usps_data(train=True)
            
            # Show source samples
            st.write("**Source Domain (MNIST):**")
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for i in range(5):
                axes[i].imshow(source_data[i].squeeze(), cmap='gray')
                axes[i].set_title(f"Label: {source_targets[i].item()}")
                axes[i].axis('off')
            st.pyplot(fig)
            
            # Show target samples
            st.write("**Target Domain (USPS):**")
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for i in range(5):
                axes[i].imshow(target_data[i].squeeze(), cmap='gray')
                axes[i].set_title(f"Label: {target_targets[i].item()}")
                axes[i].axis('off')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading sample images: {e}")

with tab2:
    st.header("Model Comparison")
    
    if st.button("🚀 Train and Compare Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Set up
                set_seed(42)
                device = get_device()
                
                # Load data
                source_data, source_targets = load_mnist_data(train=True)
                target_data, target_targets = load_usps_data(train=True)
                
                # Create models
                models = {
                    "Source Only": SourceOnlyModel(feature_dim=feature_dim, dropout=dropout),
                    "Target Only": TargetOnlyModel(feature_dim=feature_dim, dropout=dropout),
                    "Adversarial": AdversarialDomainAdaptationModel(feature_dim=feature_dim, dropout=dropout),
                    "CORAL": CORALModel(feature_dim=feature_dim, dropout=dropout),
                }
                
                # Simple training simulation (for demo purposes)
                results = {}
                
                for name, model in models.items():
                    model.to(device)
                    
                    # Simple training simulation
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    # Simulate training with a few batches
                    model.train()
                    for epoch in range(min(5, num_epochs)):  # Limit for demo
                        # Sample batches
                        source_batch = source_data[:batch_size].to(device)
                        source_labels = source_targets[:batch_size].to(device)
                        target_batch = target_data[:batch_size].to(device)
                        
                        optimizer.zero_grad()
                        
                        if name == "Source Only":
                            logits = model(source_batch)
                            loss = criterion(logits, source_labels)
                        elif name == "Target Only":
                            logits = model(target_batch)
                            loss = criterion(logits, target_targets[:batch_size].to(device))
                        elif name == "Adversarial":
                            logits, domain_logits = model(source_batch)
                            loss = criterion(logits, source_labels)
                        elif name == "CORAL":
                            logits = model(source_batch)
                            loss = criterion(logits, source_labels)
                        
                        loss.backward()
                        optimizer.step()
                    
                    # Evaluate
                    model.eval()
                    with torch.no_grad():
                        # Source accuracy
                        source_logits = model(source_data[:1000].to(device))
                        if isinstance(source_logits, tuple):
                            source_logits = source_logits[0]
                        source_pred = torch.argmax(source_logits, dim=1)
                        source_acc = (source_pred == source_targets[:1000].to(device)).float().mean()
                        
                        # Target accuracy
                        target_logits = model(target_data[:1000].to(device))
                        if isinstance(target_logits, tuple):
                            target_logits = target_logits[0]
                        target_pred = torch.argmax(target_logits, dim=1)
                        target_acc = (target_pred == target_targets[:1000].to(device)).float().mean()
                    
                    results[name] = {
                        "Source Accuracy": source_acc.item(),
                        "Target Accuracy": target_acc.item(),
                        "Parameters": sum(p.numel() for p in model.parameters()),
                    }
                
                # Display results
                st.success("Training completed!")
                
                # Results table
                st.subheader("Results")
                
                # Create DataFrame-like display
                col_names = ["Model", "Source Accuracy", "Target Accuracy", "Parameters"]
                rows = []
                for name, metrics in results.items():
                    rows.append([
                        name,
                        f"{metrics['Source Accuracy']:.3f}",
                        f"{metrics['Target Accuracy']:.3f}",
                        f"{metrics['Parameters']:,}"
                    ])
                
                # Display as table
                st.table(pd.DataFrame(rows, columns=col_names))
                
                # Visualization
                st.subheader("Performance Comparison")
                
                models_list = list(results.keys())
                source_accs = [results[m]["Source Accuracy"] for m in models_list]
                target_accs = [results[m]["Target Accuracy"] for m in models_list]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Source Accuracy", x=models_list, y=source_accs))
                fig.add_trace(go.Bar(name="Target Accuracy", x=models_list, y=target_accs))
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Model",
                    yaxis_title="Accuracy",
                    barmode="group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during training: {e}")
                st.exception(e)

with tab3:
    st.header("Domain Analysis")
    
    st.subheader("Domain Shift Visualization")
    
    if st.button("📊 Analyze Domain Shift"):
        with st.spinner("Analyzing domain shift..."):
            try:
                # Load data
                source_data, source_targets = load_mnist_data(train=True)
                target_data, target_targets = load_usps_data(train=True)
                
                # Sample data for visualization
                source_sample = source_data[:1000]
                target_sample = target_data[:1000]
                
                # Resize target to match source
                target_resized = F.interpolate(target_sample, size=(28, 28), mode='bilinear', align_corners=False)
                
                # Compute statistics
                source_mean = source_sample.mean()
                source_std = source_sample.std()
                target_mean = target_resized.mean()
                target_std = target_resized.std()
                
                # Display statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Source Mean", f"{source_mean:.3f}")
                    st.metric("Source Std", f"{source_std:.3f}")
                
                with col2:
                    st.metric("Target Mean", f"{target_mean:.3f}")
                    st.metric("Target Std", f"{target_std:.3f}")
                
                # Distribution comparison
                st.subheader("Pixel Value Distributions")
                
                source_pixels = source_sample.flatten().numpy()
                target_pixels = target_resized.flatten().numpy()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=source_pixels, name="Source (MNIST)", opacity=0.7))
                fig.add_trace(go.Histogram(x=target_pixels, name="Target (USPS)", opacity=0.7))
                
                fig.update_layout(
                    title="Pixel Value Distribution Comparison",
                    xaxis_title="Pixel Value",
                    yaxis_title="Frequency",
                    barmode="overlay"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Class distribution
                st.subheader("Class Distribution")
                
                source_classes = torch.bincount(source_targets[:1000])
                target_classes = torch.bincount(target_targets[:1000])
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(10)), y=source_classes.numpy(), name="Source"))
                fig.add_trace(go.Bar(x=list(range(10)), y=target_classes.numpy(), name="Target"))
                
                fig.update_layout(
                    title="Class Distribution Comparison",
                    xaxis_title="Digit Class",
                    yaxis_title="Count",
                    barmode="group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")

with tab4:
    st.header("Results and Insights")
    
    st.subheader("Key Findings")
    
    insights = [
        "**Domain Shift Impact:** Significant performance drop when applying source-only models to target domain",
        "**Adversarial Training:** Effective at learning domain-invariant features but may hurt source performance",
        "**CORAL Alignment:** Good balance between source and target performance",
        "**Target-Only Baseline:** Shows upper bound of target domain performance"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")
    
    st.subheader("Method Comparison")
    
    comparison_data = {
        "Method": ["Source Only", "Target Only", "Adversarial", "CORAL"],
        "Source Performance": ["High", "Low", "Medium", "High"],
        "Target Performance": ["Low", "High", "High", "Medium"],
        "Training Complexity": ["Low", "Low", "High", "Medium"],
        "Domain Invariance": ["Low", "N/A", "High", "Medium"]
    }
    
    st.table(pd.DataFrame(comparison_data))
    
    st.subheader("Best Practices")
    
    best_practices = [
        "Start with source-only baseline to establish performance floor",
        "Use target-only as upper bound for target domain performance",
        "Consider computational cost vs. performance trade-offs",
        "Validate on both source and target domains",
        "Monitor for catastrophic forgetting in adversarial methods"
    ]
    
    for practice in best_practices:
        st.markdown(f"• {practice}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Author:</strong> kryptologyst | <strong>GitHub:</strong> <a href="https://github.com/kryptologyst">https://github.com/kryptologyst</a></p>
    <p><em>This demo is for educational and research purposes only.</em></p>
</div>
""", unsafe_allow_html=True)
