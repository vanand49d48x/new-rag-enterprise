#!/bin/bash

# RAG Optimization Script
# Comprehensive optimization for TinyLlama and Qwen2.5-3B
# Integrates with existing infrastructure

echo "🎯 RAG Optimization System"
echo "=========================="

# Function to show available models
show_models() {
    echo ""
    echo "📋 Available Models for Optimization:"
    echo "====================================="
    echo "  1. qwen2.5-3b    - Balanced performance and quality"
    echo "  2. tinyllama      - Fast inference, smaller model"
    echo ""
}

# Function to show optimization types
show_optimization_types() {
    echo ""
    echo "🔧 Optimization Types:"
    echo "====================="
    echo "  quality_focus  - Maximum quality, slower response"
    echo "  speed_focus    - Fast response, basic quality"
    echo "  balanced       - Best balance of quality and speed"
    echo ""
}

# Function to run optimization
run_optimization() {
    local model=$1
    local optimization_type=$2
    
    echo ""
    echo "🔧 Running optimization for $model ($optimization_type)..."
    
    # Check if system is running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "❌ System not running. Starting services..."
        ./start.sh
        sleep 30
    fi
    
    # Run the optimization test
    cd test/quality
    python3 test_rag_improvement.py optimize "$model"
    cd ../..
    
    echo "✅ Optimization completed for $model"
}

# Function to apply best configuration
apply_best_config() {
    local model=$1
    
    echo ""
    echo "🏆 Applying best configuration for $model..."
    
    # Run optimization to find best config
    cd test/quality
    python3 test_rag_improvement.py optimize "$model" > /tmp/optimization_results.txt 2>&1
    
    # Extract best configuration from results
    best_config=$(grep "BEST CONFIGURATION:" /tmp/optimization_results.txt | awk '{print $3}')
    
    if [ -n "$best_config" ]; then
        echo "📋 Best configuration found: $best_config"
        
        # Apply the configuration
        ./apply_config.sh optimized "config_${model//./}_${best_config}"
        
        echo "✅ Applied best configuration: $best_config"
    else
        echo "❌ Could not determine best configuration"
    fi
    
    cd ../..
}

# Function to quick test current model
quick_test() {
    local model=$1
    
    echo ""
    echo "⚡ Quick test for $model..."
    
    # Check current model
    current_model=$(curl -s http://localhost:8000/system | grep -o 'model.*' | cut -d' ' -f2)
    echo "Current model: $current_model"
    
    # Run a quick test
    cd test/quality
    python test_rag_improvement.py test "$model" "balanced"
    cd ../..
}

# Function to compare models
compare_models() {
    echo ""
    echo "📊 Comparing TinyLlama vs Qwen2.5-3B..."
    
    models=("tinyllama" "qwen2.5-3b")
    
    for model in "${models[@]}"; do
        echo ""
        echo "🔧 Testing $model..."
        
        # Switch to model
        ./switch_model.sh "$model"
        sleep 30
        
        # Run quick test
        cd test/quality
        python3 test_rag_improvement.py test "$model" "balanced"
        cd ../..
    done
    
    echo ""
    echo "📊 Model comparison completed!"
}

# Function to interactive mode
interactive_mode() {
    echo ""
    echo "🎯 Interactive RAG Optimization"
    echo "=============================="
    
    while true; do
        echo ""
        echo "Options:"
        echo "  1. Show available models"
        echo "  2. Run optimization for specific model"
        echo "  3. Apply best configuration"
        echo "  4. Quick test current model"
        echo "  5. Compare all models"
        echo "  6. Exit"
        echo ""
        
        read -p "Select option (1-6): " choice
        
        case $choice in
            1)
                show_models
                ;;
            2)
                show_models
                read -p "Select model (1-2): " model_choice
                show_optimization_types
                read -p "Select optimization type: " opt_type
                
                case $model_choice in
                    1) model="qwen2.5-3b" ;;
                    2) model="tinyllama" ;;
                    *) echo "❌ Invalid choice"; continue ;;
                esac
                
                run_optimization "$model" "$opt_type"
                ;;
            3)
                show_models
                read -p "Select model (1-2): " model_choice
                
                case $model_choice in
                    1) model="qwen2.5-3b" ;;
                    2) model="tinyllama" ;;
                    *) echo "❌ Invalid choice"; continue ;;
                esac
                
                apply_best_config "$model"
                ;;
            4)
                show_models
                read -p "Select model (1-2): " model_choice
                
                case $model_choice in
                    1) model="qwen2.5-3b" ;;
                    2) model="tinyllama" ;;
                    *) echo "❌ Invalid choice"; continue ;;
                esac
                
                quick_test "$model"
                ;;
            5)
                compare_models
                ;;
            6)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid option"
                ;;
        esac
    done
}

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  optimize <model> <type>  - Run optimization for specific model and type"
    echo "  apply-best <model>        - Apply best configuration for model"
    echo "  quick-test <model>        - Quick test for model"
    echo "  compare                   - Compare all models"
    echo "  interactive               - Interactive mode"
    echo "  models                    - Show available models"
    echo "  help                      - Show this help"
    echo ""
    echo "Models:"
    echo "  qwen2.5-3b               - Qwen2.5-3B model"
    echo "  tinyllama                - TinyLlama model"
    echo ""
    echo "Optimization Types:"
    echo "  quality_focus            - Maximum quality"
    echo "  speed_focus              - Maximum speed"
    echo "  balanced                 - Balanced approach"
    echo ""
    echo "Examples:"
    echo "  $0 optimize qwen2.5-3b balanced"
    echo "  $0 apply-best tinyllama"
    echo "  $0 interactive"
    echo ""
}

# Main script logic
case "${1:-interactive}" in
    "optimize")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "❌ Usage: $0 optimize <model> <type>"
            exit 1
        fi
        run_optimization "$2" "$3"
        ;;
    "apply-best")
        if [ -z "$2" ]; then
            echo "❌ Usage: $0 apply-best <model>"
            exit 1
        fi
        apply_best_config "$2"
        ;;
    "quick-test")
        if [ -z "$2" ]; then
            echo "❌ Usage: $0 quick-test <model>"
            exit 1
        fi
        quick_test "$2"
        ;;
    "compare")
        compare_models
        ;;
    "interactive")
        interactive_mode
        ;;
    "models")
        show_models
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 