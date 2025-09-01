# 🚀 GitHub Setup Script for X13 Seasonal Adjustment Library (PowerShell)
# This script will help you publish your library to GitHub

Write-Host "🚀 X13 Seasonal Adjustment - GitHub Setup" -ForegroundColor Blue
Write-Host "========================================"
Write-Host ""

Write-Host "Step 1: Checking current status..." -ForegroundColor Blue
Write-Host "Current directory: $(Get-Location)"
Write-Host "Git status:"
git status --short
Write-Host ""

Write-Host "Step 2: Setting up GitHub remote..." -ForegroundColor Blue
Write-Host "Adding GitHub repository as remote..."

# Add the GitHub remote (user: Gardash023)
git remote add origin https://github.com/Gardash023/x13-seasonal-adjustment.git

Write-Host "✅ GitHub remote added successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Preparing to push to GitHub..." -ForegroundColor Blue
Write-Host "Setting main branch..."
git branch -M main

Write-Host "⚠️  IMPORTANT: You need to create the GitHub repository first!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Please follow these steps:"
Write-Host "1. Go to https://github.com/new"
Write-Host "2. Repository name: x13-seasonal-adjustment" 
Write-Host "3. Description: Professional X13-ARIMA-SEATS seasonal adjustment for Python"
Write-Host "4. Select: ✅ Public"
Write-Host "5. Select: ✅ MIT License" 
Write-Host "6. Click 'Create repository'"
Write-Host ""

$response = Read-Host "Have you created the GitHub repository? (y/n)"

if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "Step 4: Pushing to GitHub..." -ForegroundColor Blue
    
    # Push to GitHub
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🎉 Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your repository is now available at:"
        Write-Host "https://github.com/Gardash023/x13-seasonal-adjustment"
        Write-Host ""
        
        Write-Host "Next steps:" -ForegroundColor Blue
        Write-Host "1. ⭐ Star your repository to show it's active"
        Write-Host "2. 📝 Enable GitHub Pages for documentation" 
        Write-Host "3. 🔧 Enable Discussions for community engagement"
        Write-Host "4. 📊 Add topics: python, time-series, seasonal-adjustment, statistics"
        Write-Host "5. 🚀 Create your first release"
        Write-Host ""
        
        Write-Host "Optional: Publish to PyPI" -ForegroundColor Yellow
        Write-Host "To make your library installable with 'pip install':"
        Write-Host "1. Create account at https://pypi.org/"
        Write-Host "2. Run: make publish"
        Write-Host ""
        
    } else {
        Write-Host "❌ Failed to push to GitHub" -ForegroundColor Red
        Write-Host "Please check:"
        Write-Host "1. Repository exists on GitHub"
        Write-Host "2. You have write access"
        Write-Host "3. Your GitHub credentials are set up"
    }
} else {
    Write-Host "Please create the GitHub repository first, then run this script again." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎊 Your X13 Seasonal Adjustment library is ready for the world!" -ForegroundColor Green
