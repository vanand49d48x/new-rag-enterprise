# 🎨 Frontend Documentation

This folder contains all frontend files for the Medical RAG System, organized by purpose.

## 📁 Folder Structure

```
frontend/
├── pages/               # HTML pages and interfaces
│   ├── clean_template.html
│   ├── client_dashboard.html
│   ├── client_demo_interface.html
│   └── debug_upload.html
├── assets/              # Static assets (CSS, JS, images)
├── components/          # Reusable frontend components
└── README.md           # This file
```

## 🎯 Page Descriptions

### **pages/clean_template.html**
- **Purpose**: Clean, minimal template for new interfaces
- **Use Case**: Starting point for new frontend development
- **Features**: Basic HTML structure with modern styling

### **pages/client_dashboard.html**
- **Purpose**: Main client dashboard interface
- **Use Case**: Primary user interface for medical RAG system
- **Features**: Chat interface, document upload, search functionality

### **pages/client_demo_interface.html**
- **Purpose**: Demo interface for client presentations
- **Use Case**: Showcasing system capabilities to clients
- **Features**: Interactive demo with sample medical queries

### **pages/debug_upload.html**
- **Purpose**: Debug interface for document upload testing
- **Use Case**: Development and testing of upload functionality
- **Features**: File upload testing, error reporting

## 🚀 Quick Access Commands

### **Open Pages in Browser**
```bash
# Open client dashboard
open frontend/pages/client_dashboard.html

# Open demo interface
open frontend/pages/client_demo_interface.html

# Open debug upload
open frontend/pages/debug_upload.html
```

### **Serve Frontend Locally**
```bash
# Using Python HTTP server
cd frontend && python -m http.server 8080

# Using Node.js (if available)
cd frontend && npx serve pages
```

## 🎨 Frontend Development

### **Adding New Pages**
1. Create new HTML file in `pages/` directory
2. Use `clean_template.html` as starting point
3. Follow existing naming conventions
4. Update this README with page description

### **Styling Guidelines**
- Use modern CSS frameworks (Bootstrap, Tailwind)
- Maintain consistent color scheme
- Ensure responsive design
- Follow accessibility guidelines

### **JavaScript Integration**
- Keep scripts modular and reusable
- Use modern ES6+ syntax
- Implement proper error handling
- Follow API integration patterns

## 🔗 API Integration

### **Backend API Endpoints**
- **Chat**: `POST /chat` - Medical query processing
- **Search**: `POST /search` - Document search
- **Upload**: `POST /upload` - Document ingestion
- **Health**: `GET /health` - System status

### **Example API Calls**
```javascript
// Chat API call
fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'What are diabetes symptoms?' })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📱 Responsive Design

### **Breakpoints**
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### **Design Principles**
- Mobile-first approach
- Touch-friendly interfaces
- Readable typography
- Fast loading times

## 🧪 Testing

### **Manual Testing**
```bash
# Test all pages load correctly
for file in frontend/pages/*.html; do
    echo "Testing: $file"
    open "$file"
done
```

### **Browser Compatibility**
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## 🔧 Development Setup

### **Local Development**
1. **Clone repository**: `git clone [repo-url]`
2. **Navigate to frontend**: `cd frontend`
3. **Start local server**: `python -m http.server 8080`
4. **Open browser**: `http://localhost:8080/pages/`

### **File Organization**
- **HTML files**: `pages/` directory
- **CSS files**: `assets/css/` directory
- **JavaScript files**: `assets/js/` directory
- **Images**: `assets/images/` directory

## 📋 Page Checklist

### **Before Deployment**
- [ ] All pages load without errors
- [ ] Responsive design works on all devices
- [ ] API integration tested
- [ ] Performance optimized
- [ ] Accessibility verified
- [ ] Cross-browser compatibility tested

### **Content Review**
- [ ] Medical terminology accurate
- [ ] User interface intuitive
- [ ] Error messages helpful
- [ ] Loading states implemented
- [ ] Success feedback clear

## 🎯 Best Practices

### **HTML Structure**
- Use semantic HTML5 elements
- Include proper meta tags
- Implement proper heading hierarchy
- Add alt text for images

### **CSS Organization**
- Use consistent naming conventions
- Implement responsive design
- Optimize for performance
- Follow BEM methodology

### **JavaScript Guidelines**
- Use modern ES6+ features
- Implement proper error handling
- Follow async/await patterns
- Keep functions small and focused

## 🔄 Version Control

### **File Naming**
- Use kebab-case for file names
- Include version numbers in comments
- Document major changes in README

### **Git Workflow**
```bash
# Add new frontend files
git add frontend/

# Commit changes
git commit -m "Add new frontend page: [description]"

# Push to repository
git push origin main
```

---

**Frontend Version**: v1.0
**Last Updated**: $(date)
**Framework**: HTML5, CSS3, JavaScript ES6+ 