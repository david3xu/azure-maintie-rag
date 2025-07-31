# Azure Universal RAG with Intelligent Agents - Claude Code Guide

## Project Overview
This is a Revolutionary Universal RAG system combining tri-modal search (Vector + Graph + GNN) with intelligent agents for zero-configuration deployment across any domain.

## Key Architectural Principles
- **Data-Driven Everything**: No hardcoded values, learn from actual data
- **Universal Truth**: No fake values or placeholders 
- **Zero Configuration**: Works with any domain from raw text data
- **Tri-Modal Unity**: Vector + Knowledge Graph + GNN working together
- **Agent Intelligence**: Dynamic tool discovery and multi-step reasoning

## Current Implementation Status
- **Phase**: Phase 1 (Foundation Architecture) - Weeks 1-2
- **Branch**: feature/universal-rag-agents-implementation
- **Architecture Health**: 6.5/10 (critical issues identified)

## Critical Issues to Fix FIRST
1. **Global DI Anti-Pattern** in `backend/api/dependencies.py` (lines 18-23)
2. **Direct Service Instantiation** in `backend/api/endpoints/unified_search_endpoint.py` (line 76)
3. **API Endpoint Overlap** - 3 query endpoints doing same functionality

## Key Files and Their Purpose
- `PROJECT_ARCHITECTURE.md`: Complete architectural vision and design rules
- `CODING_STANDARDS.md`: Mandatory coding rules (data-driven, no fake data)
- `IMPLEMENTATION_ROADMAP.md`: 12-week implementation plan

## Performance Targets
- Simple queries: <1 second
- Complex agent reasoning: <3 seconds
- Retrieval accuracy: 85-95% baseline, targeting 95-98% with agents

## Never Do
- Hardcode domain-specific logic or entity types
- Return fake/placeholder data instead of real results
- Use synchronous operations for I/O
- Create multiple APIs for the same functionality
- Bypass service layer abstractions

## Always Do
- Learn patterns from actual text data
- Use proper dependency injection with Depends()
- Implement async/await for all operations
- Follow the 6 architectural design rules
- Maintain sub-3-second response times