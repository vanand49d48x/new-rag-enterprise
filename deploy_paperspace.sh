#!/bin/bash
# Paperspace C7 Deployment Script
# Deploys Enterprise RAG System on Paperspace C7 (12 CPU, 30GB RAM)

set -e

echo "🚀 Enterprise RAG System - Paperspace C7 Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check system requirements
check_system() {
    echo -e "${BLUE}📋 Checking system requirements...${NC}"
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 8 ]; then
        echo -e "${RED}❌ Insufficient CPU cores: $CPU_CORES (minimum 8 required)${NC}"
        exit 1
    fi
    
    # Check memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 16 ]; then
        echo -e "${RED}❌ Insufficient memory: ${TOTAL_MEM}GB (minimum 16GB required)${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ System requirements met: $CPU_CORES cores, ${TOTAL_MEM}GB RAM${NC}"
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}🔧 Installing system dependencies...${NC}"
    
    # Update system
    sudo apt update -y
    
    # Install Python and pip
    sudo apt install -y python3 python3-pip python3-venv
    
    # Install Docker (if not already installed)
    if ! command -v docker &> /dev/null; then
        echo -e "${BLUE}📦 Installing Docker...${NC}"
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Install Docker Compose (if not already installed)
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${BLUE}📦 Installing Docker Compose...${NC}"
        sudo apt install -y docker-compose-plugin
    fi
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Function to setup Python environment
setup_python_env() {
    echo -e "${BLUE}🐍 Setting up Python environment...${NC}"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}✅ Python environment ready${NC}"
}

# Function to download models
download_models() {
    echo -e "${BLUE}📥 Downloading models...${NC}"
    
    mkdir -p models
    
    # Download Qwen2.5-3B if not present
    if [ ! -f "models/qwen2.5-3b-instruct-q4_k_m.gguf" ]; then
        echo -e "${BLUE}📥 Downloading Qwen2.5-3B...${NC}"
        wget -O models/qwen2.5-3b-instruct-q4_k_m.gguf \
            "https://huggingface.co/TheBloke/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    fi
    
    # Download Qwen2-7B if not present
    if [ ! -f "models/qwen2-7b-instruct-q4_k_m.gguf" ]; then
        echo -e "${BLUE}📥 Downloading Qwen2-7B...${NC}"
        wget -O models/qwen2-7b-instruct-q4_k_m.gguf \
            "https://huggingface.co/TheBloke/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf"
    fi
    
    echo -e "${GREEN}✅ Models downloaded${NC}"
}

# Function to configure for C7
configure_c7() {
    echo -e "${BLUE}⚙️  Configuring for C7 (12 CPU, 30GB RAM)...${NC}"
    
    # Update config.yaml for C7
    if [ -f "config.yaml" ]; then
        # Backup original config
        cp config.yaml config.yaml.backup
        
        # Update performance settings for C7
        sed -i 's/threads: 12/threads: 12/' config.yaml
        sed -i 's/max_memory: "24GB"/max_memory: "24GB"/' config.yaml
        sed -i 's/batch_size: 1024/batch_size: 1024/' config.yaml
        
        echo -e "${GREEN}✅ C7 configuration applied${NC}"
    fi
}

# Function to start services
start_services() {
    echo -e "${BLUE}🚀 Starting Enterprise RAG System...${NC}"
    
    # Make scripts executable
    chmod +x start.sh switch_model.sh test_models.py
    
    # Start services
    ./start.sh
    
    echo -e "${GREEN}✅ Services started${NC}"
}

# Function to wait for services
wait_for_services() {
    echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
    
    # Wait for LLM server
    echo -e "${BLUE}⏳ Waiting for LLM server...${NC}"
    for i in {1..30}; do
        if curl -f -s "http://localhost:8080/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ LLM server ready${NC}"
            break
        fi
        echo "   Attempt $i/30 - waiting 10 seconds..."
        sleep 10
    done
    
    # Wait for backend API
    echo -e "${BLUE}⏳ Waiting for backend API...${NC}"
    for i in {1..30}; do
        if curl -f -s "http://localhost:8000/" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Backend API ready${NC}"
            break
        fi
        echo "   Attempt $i/30 - waiting 10 seconds..."
        sleep 10
    done
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "${BLUE}🔗 Service URLs:${NC}"
    echo -e "   • Backend API: http://localhost:8000"
    echo -e "   • LLM Server: http://localhost:8080"
    echo -e "   • Vector DB: http://localhost:6334"
    echo -e "   • Grafana: http://localhost:3000 (admin/admin)"
    echo -e "   • Prometheus: http://localhost:9090"
    
    echo -e "${BLUE}🧪 Testing Commands:${NC}"
    echo -e "   • Test LLM: curl -X POST http://localhost:8080/completion -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 10}'"
    echo -e "   • Switch Model: ./switch_model.sh qwen25_3b"
    echo -e "   • Run Tests: python3 test_models.py"
}

# Main deployment function
main() {
    echo -e "${BLUE}🚀 Starting Paperspace C7 deployment...${NC}"
    
    check_system
    install_dependencies
    setup_python_env
    download_models
    configure_c7
    start_services
    wait_for_services
    show_status
    
    echo -e "${GREEN}🎉 Enterprise RAG System deployed successfully on Paperspace C7!${NC}"
    echo -e "${YELLOW}💡 Don't forget to logout and login again for Docker group to take effect${NC}"
}

# Run main function
main "$@" 