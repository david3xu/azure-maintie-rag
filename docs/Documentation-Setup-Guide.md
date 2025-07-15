# Documentation Setup Guide

## 📝 Enhanced VSCode Markdown Experience

This guide explains the enhanced documentation setup for the MaintIE Enhanced RAG project, providing a professional markdown reading and writing experience.

---

## 🚀 Quick Setup

### From Project Root
```bash
make docs-setup    # Setup VSCode environment with extensions
make docs-status   # Show documentation setup status
make docs-preview  # Open markdown preview (if VSCode CLI available)
```

### From Backend Directory
```bash
cd backend
make docs-setup    # Setup VSCode environment with extensions
make docs-status   # Show documentation setup status
make docs-preview  # Open markdown preview (if VSCode CLI available)
```

---

## 🛠️ What's Configured

### VSCode Extensions
The setup automatically configures these extensions for optimal markdown experience:

**Markdown Extensions:**
- **Markdown All in One** - All-in-one markdown extension
- **Markdown Preview Enhanced** - Enhanced markdown preview with custom CSS
- **Markdown Mermaid** - Mermaid diagram support
- **Markdownlint** - Markdown linting and formatting

**Development Extensions:**
- **Python** - Python language support
- **Black Formatter** - Python code formatting
- **Pylint** - Python linting
- **JSON** - JSON file support
- **YAML** - YAML file support

### VSCode Settings
Custom settings for optimal markdown preview:

```json
{
  "markdown.preview.fontSize": 14,
  "markdown.preview.lineHeight": 1.6,
  "markdown.preview.fontFamily": "system-ui, -apple-system, sans-serif",
  "markdown.preview.breaks": false,
  "markdown.preview.typographer": true,
  "markdown.extension.toc.levels": "1..6",
  "markdown.extension.preview.autoShowPreviewToSide": true,
  "markdown.preview.maxWidth": 1400
}
```

### Custom CSS Styling
Professional styling optimized for research-level ML/AI documentation:

- **Wider Layout**: 1400px max-width for complex code blocks
- **Professional Typography**: System fonts with coding fonts
- **Syntax Highlighting**: Enhanced code block styling
- **Research Focus**: Special styling for ML/AI content
- **Tree Structures**: Highlighted ASCII art trees
- **Technology Stacks**: Color-coded stack visualizations

---

## 🔧 SSH Development (Azure ML)

### Recommended Setup
For the best experience in SSH environments like Azure ML:

1. **Install VSCode Remote-SSH Extension** locally
2. **Connect to your SSH server**:
   ```
   ssh azureuser@your-azure-ml-ip
   ```
3. **All extensions auto-install** when you connect
4. **Use `Ctrl+Shift+V`** for side-by-side markdown preview

### Alternative: VSCode Server
If you prefer to install VSCode Server on the remote machine:
```bash
# Install VSCode Server
curl -fsSL https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64 -o vscode-cli.tar.gz
tar -xzf vscode-cli.tar.gz
./code --install-extension yzhang.markdown-all-in-one
```

---

## 📖 Using the Enhanced Markdown

### Code Blocks with Language Tags
```markdown
```python
# Your Python code
def process_query_structured():
    # Implementation
    pass
```

```typescript
// Your React code
const handleSubmit = async (e: React.FormEvent) => {
  // Implementation
};
```
```

### Technology Stack Trees
```markdown
```
Frontend Stack:
├─ React 19.1.0 + TypeScript
├─ Vite 7.0.4 (build tool)
├─ axios 1.10.0 (HTTP client)
└─ CSS custom styling

Backend Stack:
├─ FastAPI + uvicorn
├─ Azure OpenAI integration
├─ FAISS 1.7.4 vector search
└─ NetworkX 3.2.0 graph processing
```
```

### Professional Flow Diagrams
```markdown
```
User Input → React Frontend → FastAPI Backend → AI Processing → Response
     ↓             ↓              ↓               ↓           ↓
"pump failure"  handleSubmit()  POST /api/      RAG pipeline  JSON response
```
```

---

## 🎨 Custom CSS Features

### Special Styling for ML/AI Content
- **Highlighted sections** for important information
- **Color-coded technology stacks**
- **Enhanced code block readability**
- **Professional typography**

### Responsive Design
- **Wide layout** for complex documentation
- **Readable font sizes** and line heights
- **Proper spacing** and margins
- **Dark/light mode** compatibility

---

## 🔄 Maintenance

### Updating Extensions
To update the extension list:
1. Edit `.vscode/extensions.json`
2. Run `make docs-setup` to apply changes

### Updating CSS
To modify the styling:
1. Edit `.vscode/markdown.css`
2. Restart VSCode or reload the window

### Updating Settings
To change VSCode settings:
1. Edit `.vscode/settings.json`
2. Settings apply immediately

---

## 📋 Troubleshooting

### Common Issues

**VSCode CLI not available:**
- Use VSCode Remote-SSH extension instead
- Or install VSCode Server on remote machine

**Extensions not installing:**
- Check internet connection
- Restart VSCode
- Manually install extensions from marketplace

**CSS not applying:**
- Restart VSCode
- Check if markdown preview is using custom CSS
- Verify `.vscode/markdown.css` exists

### Getting Help
- Check `make docs-status` for setup verification
- Review VSCode extension marketplace for individual extension issues
- Consult VSCode Remote-SSH documentation for SSH-specific issues

---

## 🎯 Best Practices

### Writing Documentation
1. **Use proper language tags** in code blocks
2. **Structure with clear headings** (H1, H2, H3)
3. **Include technology stack trees** for clarity
4. **Use flow diagrams** for complex processes
5. **Add syntax highlighting** to all code examples

### Maintaining Documentation
1. **Update when architecture changes**
2. **Keep extension list current**
3. **Test markdown preview regularly**
4. **Use consistent formatting**

---

**Last Updated**: 2024
**Maintainer**: Development Team
**Version**: 1.0