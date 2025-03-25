import json
import logging
from typing import Dict, List

# Use relative imports for modules in the same package
from .router import (
    RouterOperator, 
    RouterInput, 
    RouterOutput,
    ModelPreference
)

# For modules outside the current package, you might need to use a try/except pattern
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Mock LM for testing
class MockLMModule(LMModule):
    """Mock LM module that returns predefined responses."""
    
    def __init__(self, model_name: str, mock_responses: Dict[str, str] = None):
        """Initialize with a model name and optional mock responses.
        
        Args:
            model_name: Name of the mocked model.
            mock_responses: Dictionary mapping query substrings to responses.
        """
        # Use a simple config to avoid actual API calls
        super().__init__(
            config=LMModuleConfig(
                id=f"mock:{model_name}", # no model specified here yet
                temperature=0,
                simulate_api=True  # Important: prevents actual API calls
            )
        )
        self.model_name = model_name
        self.mock_responses = mock_responses or {}
        
    def forward(self, prompt: str, **kwargs) -> str:
        """Return mock response based on the prompt.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            A predefined response based on the prompt.
        """
        # For router model, return a mock ranking response
        if "You are a specialized router" in prompt:
            # Router prompts, change if needed
            if "code" in prompt.lower() or "function" in prompt.lower() or "programming" in prompt.lower():
                return json.dumps([
                    {"model_name": "code_specialist", "score": 0.92, "reasoning": "This is a programming question."},
                    {"model_name": "general_purpose_model", "score": 0.65, "reasoning": "Can handle general queries."},
                    {"model_name": "math_specialist", "score": 0.45, "reasoning": "Not primarily a math question."},
                    {"model_name": "creative_writer", "score": 0.21, "reasoning": "Not a creative task."}
                ])
            elif "math" in prompt.lower() or "equation" in prompt.lower() or "calculate" in prompt.lower():
                return json.dumps([
                    {"model_name": "math_specialist", "score": 0.89, "reasoning": "This is a math question."},
                    {"model_name": "general_purpose_model", "score": 0.70, "reasoning": "Can handle general queries."},
                    {"model_name": "code_specialist", "score": 0.55, "reasoning": "May involve some logical reasoning."},
                    {"model_name": "creative_writer", "score": 0.18, "reasoning": "Not a creative task."}
                ])
            elif "story" in prompt.lower() or "poem" in prompt.lower() or "creative" in prompt.lower():
                return json.dumps([
                    {"model_name": "creative_writer", "score": 0.94, "reasoning": "This is a creative task."},
                    {"model_name": "general_purpose_model", "score": 0.75, "reasoning": "Can handle general queries."},
                    {"model_name": "code_specialist", "score": 0.15, "reasoning": "Not a programming question."},
                    {"model_name": "math_specialist", "score": 0.10, "reasoning": "Not a math question."}
                ])
            else:
                return json.dumps([
                    {"model_name": "general_purpose_model", "score": 0.88, "reasoning": "General query best handled by general-purpose model."},
                    {"model_name": "code_specialist", "score": 0.35, "reasoning": "Not specifically about code."},
                    {"model_name": "math_specialist", "score": 0.30, "reasoning": "Not specifically about math."},
                    {"model_name": "creative_writer", "score": 0.40, "reasoning": "May have some creative elements."}
                ])
        
        # For specialized models, return a mock response based on their specialty
        if self.model_name == "code_specialist":
            return "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
        elif self.model_name == "math_specialist":
            return "To solve this equation, we need to use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a"
        elif self.model_name == "creative_writer":
            return "Autumn leaves dance gently down,\nA cascade of crimson and gold,\nWhispering secrets of the fading year,\nAs summer's tales unfold."
        else:
            return f"General response from {self.model_name}: This is a simulated response."


def test_router_with_different_queries():
    """Test the router with different types of queries."""
    
    # Create mock models
    mock_models = {
        "general_purpose_model": MockLMModule("general_purpose_model"),
        "code_specialist": MockLMModule("code_specialist"),
        "math_specialist": MockLMModule("math_specialist"),
        "creative_writer": MockLMModule("creative_writer"),
    }
    
    # Create a router with mock router model
    router = RouterOperator(
        models=mock_models,
        model_descriptions={
            "general_purpose_model": "Good for general questions across a wide range of topics.",
            "code_specialist": "Excels at programming tasks and code generation.",
            "math_specialist": "Specialized in mathematical problems and equations.",
            "creative_writer": "Best for creative tasks like storytelling and poetry."
        }
    )
    
    # Replace the router's LM with our mock
    router.router_lm = MockLMModule("router_model")
    
    # Test queries
    test_queries = [
        "Write a function to calculate the Fibonacci sequence in Python",
        "Solve the equation: 2x² + 3x - 5 = 0",
        "Write a poem about autumn leaves",
        "What is the capital of France?",
    ]
    
    expected_models = [
        "code_specialist",
        "math_specialist",
        "creative_writer",
        "general_purpose_model",
    ]
    
    # Run tests
    for i, query in enumerate(test_queries):
        logger.info(f"\nTesting query: {query}")
        
        # Call the router
        result = router(inputs=RouterInput(query=query))
        
        # Log the results
        logger.info(f"Selected model: {result.selected_model} (Confidence: {result.confidence:.2f})")
        logger.info(f"Response: {result.final_answer[:50]}...")
        
        # Verify it chose the expected model
        assert result.selected_model == expected_models[i], f"Expected {expected_models[i]}, got {result.selected_model}"
        
        logger.info(f"✓ Test passed for query: {query}")
    
    logger.info("\nAll tests passed!")


def test_router_with_additional_attributes():
    """Test the router with modified implementation that returns all model rankings."""
    
    # First update our RouterOutput model to include all model preferences
    RouterOutput.model_preferences = List[ModelPreference]
    
    # Create mock models
    mock_models = {
        "general_purpose_model": MockLMModule("general_purpose_model"),
        "code_specialist": MockLMModule("code_specialist"),
        "math_specialist": MockLMModule("math_specialist"),
        "creative_writer": MockLMModule("creative_writer"),
    }
    
    # Create our router
    router = RouterOperator(
        models=mock_models,
        model_descriptions={
            "general_purpose_model": "Good for general questions across a wide range of topics.",
            "code_specialist": "Excels at programming tasks and code generation.",
            "math_specialist": "Specialized in mathematical problems and equations.",
            "creative_writer": "Best for creative tasks like storytelling and poetry."
        }
    )
    
    # Replace the router's LM with our mock
    router.router_lm = MockLMModule("router_model")
    
    # Add a monkey patch to the router's forward method to return preferences
    original_forward = router.forward
    
    def forward_with_preferences(*, inputs):
        context_section = f"Context: {inputs.context}" if inputs.context else ""
        model_desc_text = "\n".join([f"- {name}: {desc}" for name, desc in router.model_descriptions.items()])
        
        routing_inputs = {
            "query": inputs.query,
            "context_section": context_section,
            "model_descriptions": model_desc_text
        }
        
        routing_prompt = router.specification.render_prompt(inputs=routing_inputs)
        routing_response = router.router_lm(prompt=routing_prompt)
        
        import json
        try:
            json_str = routing_response
            if "```json" in routing_response:
                json_str = routing_response.split("```json")[1].split("```")[0].strip()
            elif "```" in routing_response:
                json_str = routing_response.split("```")[1].split("```")[0].strip()
                
            preferences = [ModelPreference(**item) for item in json.loads(json_str)]
        except Exception as e:
            preferences = [
                ModelPreference(
                    model_name=next(iter(router.available_models.keys())),
                    score=1.0,
                    reasoning="Default selection due to parsing error."
                )
            ]
        
        selected_preference = max(preferences, key=lambda p: p.score)
        selected_model_name = selected_preference.model_name
        confidence = selected_preference.score
        
        if selected_model_name in router.available_models:
            selected_model = router.available_models[selected_model_name]
        else:
            selected_model_name = next(iter(router.available_models.keys()))
            selected_model = router.available_models[selected_model_name]
            confidence = 0.5
        
        model_response = selected_model(prompt=inputs.query)
        
        # Create output with preferences
        output = RouterOutput(
            final_answer=model_response if isinstance(model_response, str) else str(model_response),
            selected_model=selected_model_name,
            confidence=confidence
        )
        
        # Add the model preferences to the output
        output.model_preferences = preferences
        
        return output
    
    # Patch the forward method
    router.forward = forward_with_preferences
    
    # Test with a code query
    query = "Write a function to calculate the Fibonacci sequence in Python"
    logger.info(f"\nTesting query with preferences: {query}")
    
    # Call the router
    result = router(inputs=RouterInput(query=query))
    
    # Log the results
    logger.info(f"Selected model: {result.selected_model} (Confidence: {result.confidence:.2f})")
    logger.info(f"Response: {result.final_answer[:50]}...")
    
    # Log all model preferences
    logger.info("All model preferences:")
    for pref in result.model_preferences:
        logger.info(f"- {pref.model_name}: {pref.score:.2f} ({pref.reasoning})")
    
    # Verify the ranking is returned
    assert hasattr(result, 'model_preferences'), "model_preferences attribute is missing"
    assert len(result.model_preferences) > 0, "model_preferences list is empty"
    
    # Restore the original forward method
    router.forward = original_forward
    
    logger.info("✓ Test passed for query with preferences!")


if __name__ == "__main__":
    test_router_with_different_queries()
    test_router_with_additional_attributes()