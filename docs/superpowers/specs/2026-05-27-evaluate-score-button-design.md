# Evaluate Score Button - Design Spec

## Overview

Add a "Show Evaluate Score" button below the assistant's response that opens the right sidebar to display RAGAS evaluation scores when clicked.

## User Flow

1. User sends a query → receives RAG response
2. User clicks "Show Evaluate Score" button below the response
3. Right sidebar opens immediately with "Evaluating..." message and spinner
4. System runs RAGAS evaluation on the current response
5. Scores populate in the sidebar when ready

## Design Details

### Button Placement
- Located below the sources expander in the assistant's response area
- Standard Streamlit button styling

### Right Sidebar Behavior
- Uses Streamlit's built-in sidebar (right side of page)
- Width: default Streamlit sidebar width (~300px)
- Contains evaluation panel that appears on button click
- Toggled via `st.session_state`

### Evaluation Panel Content
- **Header:** "Evaluation Scores"
- **Evaluating state:** Spinner with "Evaluating..." text
- **Results state:** Three scores displayed:
  - Faithfulness: 0.XX
  - Answer Relevancy: 0.XX
  - Context Relevance: 0.XX
- Scores are 0-1 float values

### Session State Keys
- `show_eval_panel`: bool — controls sidebar visibility
- `current_eval_result`: EvalResult — stores evaluation results
- `eval_in_progress`: bool — indicates if evaluation is running

## Implementation

### Files to Modify
- `streamlit_app.py` — main UI changes

### Components
1. **Button** — `st.button("Show Evaluate Score")` below sources
2. **Sidebar section** — conditionally rendered based on session state
3. **Evaluation integration** — uses `RAGASEvaluator` on current query/answer

### Data Flow
1. Query result stored temporarily in session state
2. Button click triggers evaluation
3. `RAGASEvaluator.run_case()` runs on the stored result
4. Results stored in session state
5. Sidebar displays results

### Error Handling
- Handle evaluation failures gracefully (show error message in sidebar)
- Disable button while evaluation is in progress