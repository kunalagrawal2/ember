import json
from typing import Dict, List, ClassVar, Any, Optional, Type
from ember.core.registry.operator.base.operator_base import Operator
from ember.api.models import EmberModel
from ember.core.registry.specification.specification import Specification
from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig


# Prompt template for LLM decision router to follow -- generalize in prod
ROUTER_PROMPT_TEMPLATE = """You are a specialized router that analyzes queries and determines the most appropriate AI model to handle them.

For the following query, evaluate which model would be best suited to respond based on the query content and model capabilities.

Query: {query}
{context_section}

Available models:
{model_descriptions}

For each model, assign a score from 0.0 to 1.0 representing how well it can handle this specific query.
Also provide brief reasoning for each score.

Respond in this exact format:

[
    {
        "model_name": "Model1",
        "score": 0.X,
        "reasoning": "Brief explanation"
    },
    {
        "model_name": "Model2",
        "score": 0.Y,
        "reasoning": "Brief explanation"
    }
]

Ensure the scores accurately reflect each model's suitability for this specific query type.

"""


class RouterInput(EmberModel):
    """Input model containing the query to route to the appropriate model.
    
    Attributes:
        query: The user's question or prompt text.
        context: Optional context information for the query.
    """
    query: str
    context: Optional[str] = None


class RouterOutput(EmberModel):
    """Output model containing the routed response and metadata.
    
    Attributes:
        final_answer: The response from the selected model.
        selected_model: The name of the model that was selected.
        confidence: Confidence score for the routing decision.
        model_preferences: List of all models with their Bradley-Terry coefficients and reasoning.
    """
    final_answer: str
    selected_model: str
    confidence: float
    model_preferences: List["ModelPreference"]


class ModelPreference(EmberModel):
    """Model for representing the router's preference scores.
    
    Attributes:
        model_name: Name of the model.
        score: Bradley-Terry coefficient representing suitability for the query.
        reasoning: Explanation for why this model might be suitable.
    """
    model_name: str
    score: float
    reasoning: str


class RouterSpecification(Specification):
    """Specification for the router operator.
    
    Defines input/output models and prompt template for routing.
    """
    input_model = RouterInput
    structured_output = RouterOutput
    prompt_template = ROUTER_PROMPT_TEMPLATE


class RouterOperator(Operator[RouterInput, RouterOutput]):
    """Routes queries to the most appropriate model based on content analysis.
    
    This operator analyzes the query content and selects the most appropriate
    model from a set of available models, inspired by the Prompt to Leaderboard
    approach of predicting model performance on specific inputs.
    
    Attributes:
        specification: The input/output specification for this operator.
        router_lm: The LM module used for routing decisions.
        available_models: Dictionary of model instances keyed by name.
        model_descriptions: Dictionary of model descriptions for the router context.
    """
    
    specification: ClassVar[Specification] = RouterSpecification()
    router_lm: LMModule
    available_models: Dict[str, LMModule]
    model_descriptions: Dict[str, str]
    
    def __init__(
        self, 
        router_model_name: str = "anthropic:claude-3-5-sonnet", # can be changed to model of choice, generalize
        models: Optional[Dict[str, LMModule]] = None,
        model_descriptions: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize the router with a routing model and available target models.
        
        Args:
            router_model_name: Name of the model to use for routing decisions.
            models: Dictionary of available models keyed by name.
            model_descriptions: Descriptions of model capabilities for context.
        """
        # Initialize the router LM model
        self.router_lm = LMModule(
            config=LMModuleConfig(
                id=router_model_name,
                temperature=0.1,  # Low temperature for consistent routing
            )
        )
        
        # Initialize available models
        self.available_models = models or {}
        
        # Initialize model descriptions, can be abstracted
        self.model_descriptions = model_descriptions or {
            "general_purpose_model": "Good for general questions across a wide range of topics. Balanced capabilities.",
            "code_specialist": "Excels at programming tasks, code generation, debugging, and technical explanations.",
            "math_specialist": "Specialized in mathematical problems, equations, proofs, and numerical reasoning.",
            "creative_writer": "Best for creative tasks like storytelling, poetry, creative writing, and imaginative content."
        }
        
        # in case models are not provided
        if not models:
            for model_name in self.model_descriptions.keys():
                self.available_models[model_name] = LMModule(
                    config=LMModuleConfig(
                        id="openai:gpt-4o",  # Default model, subject to change
                        temperature=0.2, # initialized to low temp for consistency
                        persona=f"You are a {model_name.replace('_', ' ')}."
                    )
                )
    
    def forward(self, *, inputs: RouterInput) -> RouterOutput:
        """Route the query to the most appropriate model based on content analysis.
        
        Args:
            inputs: The input containing the query to route.
            
        Returns:
            The output containing the response from the selected model.
        """

        context_section = f"Context: {inputs.context}" if inputs.context else ""
        
        # Format model descriptions for the prompt
        model_desc_text = "\n".join([f"- {name}: {desc}" for name, desc in self.model_descriptions.items()])
        
        # Prepare the routing prompt
        routing_inputs = {
            "query": inputs.query,
            "context_section": context_section,
            "model_descriptions": model_desc_text
        }
        
        # Get the routing decision from the router model
        routing_prompt = self.specification.render_prompt(inputs=routing_inputs)
        routing_response = self.router_lm(prompt=routing_prompt)
        
        # Parse the routing response to extract model preferences
        # convert to more robust json extraction
        try:
            # Extract the JSON part from the response
            json_str = routing_response
            if "```json" in routing_response:
                json_str = routing_response.split("```json")[1].split("```")[0].strip()
            elif "```" in routing_response:
                json_str = routing_response.split("```")[1].split("```")[0].strip()
                
            preferences = [ModelPreference(**item) for item in json.loads(json_str)]
        except Exception as e:
            # Fallback if parsing fails
            preferences = [
                ModelPreference(
                    model_name=next(iter(self.available_models.keys())),
                    score=1.0,
                    reasoning="Default selection due to parsing error."
                )
            ]
        
        # Find the model with the highest score
        selected_preference = max(preferences, key=lambda p: p.score)
        selected_model_name = selected_preference.model_name
        confidence = selected_preference.score
        
        # Get the selected model (with fallback)
        if selected_model_name in self.available_models:
            selected_model = self.available_models[selected_model_name]
        else:
            # Fallback to first available model
            selected_model_name = next(iter(self.available_models.keys()))
            selected_model = self.available_models[selected_model_name]
            confidence = 0.5  # Lower confidence for fallback
        
        # Process the query with the selected model
        model_response = selected_model(prompt=inputs.query)
        
        # Return the structured output with full model preferences
        return RouterOutput(
            final_answer=model_response if isinstance(model_response, str) else str(model_response),
            selected_model=selected_model_name,
            confidence=confidence,
            model_preferences=preferences  # Include full list of model preferences with scores
        )