#!/usr/bin/env python3
"""
Hypermode Integration for IntelliBase
Advanced agent orchestration and workflow management
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
import requests

# Check if Hypermode is available
try:
    import hypermode
    HYPERMODE_AVAILABLE = True
except ImportError:
    print("⚠️ Hypermode not available - install with: pip install hypermode")
    HYPERMODE_AVAILABLE = False

# Observability integration
try:
    from observability import obs_manager, trace_agent_operation
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    def trace_agent_operation(**kwargs):
        def decorator(func):
            return func
        return decorator


class HypermodeIntegration:
    """Hypermode integration for advanced agent orchestration"""
    
    def __init__(self):
        self.api_key = os.getenv("HYPERMODE_API_KEY")
        self.base_url = "https://api.hypermode.ai/v1"
        self.is_available = False
        self.workspace_id = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Hypermode client"""
        if not self.api_key or self.api_key == "your_hypermode_key_here":
            print("⚠️ HYPERMODE_API_KEY not found or placeholder - Hypermode disabled")
            return
        
        try:
            # Test API connection
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Get workspace info
            response = requests.get(f"{self.base_url}/workspaces", headers=headers)
            
            if response.status_code == 200:
                workspaces = response.json()
                if workspaces:
                    self.workspace_id = workspaces[0].get("id")
                    self.is_available = True
                    print("✅ Hypermode client initialized")
                    print(f"   Workspace: {self.workspace_id}")
                else:
                    print("⚠️ No workspaces found in Hypermode account")
            else:
                print(f"⚠️ Hypermode API test failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Hypermode initialization failed: {e}")
    
    @trace_agent_operation(operation="hypermode_agent_execution")
    def execute_agent(self, agent_name: str, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a Hypermode agent"""
        
        if not self.is_available:
            return {
                "success": False,
                "error": "Hypermode not available",
                "mock": True
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "workspace_id": self.workspace_id,
                "agent_name": agent_name,
                "inputs": inputs,
                **kwargs
            }
            
            print(f"🤖 Executing Hypermode agent: {agent_name}")
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/agents/execute",
                headers=headers,
                json=payload
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # Log metrics
                if OBSERVABILITY_AVAILABLE:
                    obs_manager.log_metrics("hypermode_execution", {
                        "agent_name": agent_name,
                        "execution_time": end_time - start_time,
                        "inputs_count": len(inputs),
                        "success": True
                    })
                
                return {
                    "success": True,
                    "result": result,
                    "execution_time": end_time - start_time,
                    "agent_name": agent_name
                }
            else:
                print(f"❌ Hypermode agent execution failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "execution_time": end_time - start_time
                }
                
        except Exception as e:
            print(f"❌ Hypermode execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "mock": True
            }
    
    def create_workflow(self, workflow_name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a Hypermode workflow"""
        
        if not self.is_available:
            return {
                "success": False,
                "error": "Hypermode not available",
                "mock": True
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "workspace_id": self.workspace_id,
                "name": workflow_name,
                "steps": steps
            }
            
            response = requests.post(
                f"{self.base_url}/workflows",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 201:
                return {
                    "success": True,
                    "workflow": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "mock": True
            }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List available Hypermode agents"""
        
        if not self.is_available:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/agents",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to list agents: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error listing agents: {e}")
            return []
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace information"""
        
        if not self.is_available:
            return {"available": False}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/workspaces/{self.workspace_id}",
                headers=headers
            )
            
            if response.status_code == 200:
                return {
                    "available": True,
                    "workspace": response.json()
                }
            else:
                return {
                    "available": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test Hypermode connection"""
        return self.is_available
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "available": self.is_available,
            "workspace_id": self.workspace_id,
            "api_key_configured": bool(self.api_key and self.api_key != "your_hypermode_key_here")
        }


# Create global instance
hypermode_integration = HypermodeIntegration() 