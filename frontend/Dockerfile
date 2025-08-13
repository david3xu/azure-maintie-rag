# Multi-stage build for React frontend
FROM node:20-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the frontend
RUN npm run build

# Production stage with nginx
FROM nginx:alpine

# Copy built files to nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration for SPA routing (port 3000)
RUN echo 'server { \
  listen 3000; \
  location / { \
    root /usr/share/nginx/html; \
    index index.html index.htm; \
    try_files $uri $uri/ /index.html; \
  } \
  location /api/ { \
    proxy_pass http://backend:8000; \
    proxy_set_header Host $host; \
    proxy_set_header X-Real-IP $remote_addr; \
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    proxy_set_header X-Forwarded-Proto $scheme; \
  } \
}' > /etc/nginx/conf.d/default.conf

# Expose port 3000 (matches Container App configuration)
EXPOSE 3000

# Start nginx
CMD ["nginx", "-g", "daemon off;"]