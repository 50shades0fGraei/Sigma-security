#!/bin/bash
# deploy_vercel.sh: Deploy Sigma Security to Vercel
# Showcases bug-catching with sigma swagger

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod

echo "Sigma Security deployed to Vercel. Check URL to catch bugs online!"
