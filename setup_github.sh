#!/bin/bash

# 🚀 GitHub Setup Script for X13 Seasonal Adjustment Library
# This script will help you publish your library to GitHub

echo "🚀 X13 Seasonal Adjustment - GitHub Setup"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Step 1: Checking current status...${NC}"
echo "Current directory: $(pwd)"
echo "Git status:"
git status --short
echo ""

echo -e "${BLUE}Step 2: Setting up GitHub remote...${NC}"
echo "Adding GitHub repository as remote..."

# Add the GitHub remote (user: Gardash023)
git remote add origin https://github.com/Gardash023/x13-seasonal-adjustment.git

echo -e "${GREEN}✅ GitHub remote added successfully!${NC}"
echo ""

echo -e "${BLUE}Step 3: Preparing to push to GitHub...${NC}"
echo "Setting main branch..."
git branch -M main

echo -e "${YELLOW}⚠️  IMPORTANT: You need to create the GitHub repository first!${NC}"
echo ""
echo "Please follow these steps:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: x13-seasonal-adjustment" 
echo "3. Description: Professional X13-ARIMA-SEATS seasonal adjustment for Python"
echo "4. Select: ✅ Public"
echo "5. Select: ✅ MIT License" 
echo "6. Click 'Create repository'"
echo ""

read -p "Have you created the GitHub repository? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${BLUE}Step 4: Pushing to GitHub...${NC}"
    
    # Push to GitHub
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}🎉 Successfully pushed to GitHub!${NC}"
        echo ""
        echo "Your repository is now available at:"
        echo "https://github.com/Gardash023/x13-seasonal-adjustment"
        echo ""
        
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. ⭐ Star your repository to show it's active"
        echo "2. 📝 Enable GitHub Pages for documentation" 
        echo "3. 🔧 Enable Discussions for community engagement"
        echo "4. 📊 Add topics: python, time-series, seasonal-adjustment, statistics"
        echo "5. 🚀 Create your first release"
        echo ""
        
        echo -e "${YELLOW}Optional: Publish to PyPI${NC}"
        echo "To make your library installable with 'pip install':"
        echo "1. Create account at https://pypi.org/"
        echo "2. Run: make publish"
        echo ""
        
    else
        echo -e "${RED}❌ Failed to push to GitHub${NC}"
        echo "Please check:"
        echo "1. Repository exists on GitHub"
        echo "2. You have write access"
        echo "3. Your GitHub credentials are set up"
    fi
else
    echo -e "${YELLOW}Please create the GitHub repository first, then run this script again.${NC}"
fi

echo ""
echo -e "${GREEN}🎊 Your X13 Seasonal Adjustment library is ready for the world!${NC}"
