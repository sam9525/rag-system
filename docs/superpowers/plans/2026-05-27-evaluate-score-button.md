# Evaluate Score Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Show Evaluate Score" button to RAG responses that opens right sidebar with RAGAS evaluation scores.

**Architecture:** Single-file modification to streamlit_app.py. Session state stores current query result for evaluation. Button triggers async evaluation, sidebar displays results when complete.

**Tech Stack:** Streamlit, RAGASEvaluator, asyncio

---

## File Structure

- **Modify:** `streamlit_app.py` — all changes in this file
- **Test:** `tests/test_streamlit_app.py` — verify button and sidebar behavior (if exists, else manual test)

---

## Task 1: Add Session State Variables

**Files:**
- Modify: `streamlit_app.py:68-78` (init_session_state function)

- [ ] **Step 1: Add eval panel session state**

Locate `init_session_state()` function. Add these keys after existing session state:

```python
if "current_query_result" not in st.session_state:
    st.session_state.current_query_result = None
if "show_eval_panel" not in st.session_state:
    st.session_state.show_eval_panel = False
if "eval_result" not in st.session_state:
    st.session_state.eval_result = None
if "eval_error" not in st.session_state:
    st.session_state.eval_error = None
```

- [ ] **Step 2: Verify changes**

The init_session_state function should now have 7 session state keys total.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add eval panel session state variables"
```

---

## Task 2: Store Query Result After Generation

**Files:**
- Modify: `streamlit_app.py:198-225` (main() function query handling)

- [ ] **Step 1: Store query result in session state**

After successful query (around line 202), add before rendering sources:

```python
# Store current result for evaluation
st.session_state.current_query_result = result
st.session_state.show_eval_panel = False  # Reset panel state
st.session_state.eval_result = None
st.session_state.eval_error = None
```

Find this section in main():
```python
result = st.session_state.rag_system.query(prompt)
```

Add the storage lines right after this, before rendering the answer.

- [ ] **Step 2: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: store query result for evaluation"
```

---

## Task 3: Add "Show Evaluate Score" Button

**Files:**
- Modify: `streamlit_app.py:203-213` (after sources rendering)

- [ ] **Step 1: Add button below sources**

After `render_sources(result.sources)` call (line ~213), add:

```python
# Evaluation button
if st.button("Show Evaluate Score", key=f"eval_btn_{len(st.session_state.messages)}"):
    st.session_state.show_eval_panel = True
    st.rerun()
```

The key ensures each message has a unique button instance.

- [ ] **Step 2: Verify placement**

Button should appear after the sources expander in the assistant's response area.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Show Evaluate Score button"
```

---

## Task 4: Add Right Sidebar Evaluation Panel

**Files:**
- Modify: `streamlit_app.py` (after sidebar_config function, before main function)

- [ ] **Step 1: Create evaluation panel function**

Add this new function before `main()`:

```python
def eval_panel():
    """Render evaluation panel in right sidebar."""
    with st.sidebar:
        st.subheader("Evaluation Scores")

        if st.session_state.eval_error:
            st.error(st.session_state.eval_error)
            return

        if st.session_state.eval_result:
            result = st.session_state.eval_result
            st.success("Evaluation complete")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faithfulness", f"{result.faithfulness:.2f}")
            with col2:
                st.metric("Answer Relevancy", f"{result.answer_relevancy:.2f}")

            st.metric("Context Relevance", f"{result.context_relevance:.2f}")

            st.divider()

            if st.button("Close Panel"):
                st.session_state.show_eval_panel = False
                st.rerun()
        else:
            st.info("Click 'Show Evaluate Score' to evaluate the response")
```

- [ ] **Step 2: Call eval_panel conditionally in main**

At the end of `main()` function, before the final `if __name__` block, add:

```python
# Show eval panel if enabled
if st.session_state.show_eval_panel:
    eval_panel()
```

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add evaluation panel in sidebar"
```

---

## Task 5: Wire Up Evaluation Logic

**Files:**
- Modify: `streamlit_app.py` — add evaluator import and run logic

- [ ] **Step 1: Import RAGASEvaluator**

At the top of the file with other imports (around line 15):

```python
from src.evaluation.evaluator import RAGASEvaluator
```

- [ ] **Step 2: Modify button click handler to run evaluation**

Replace the button code from Task 3 with:

```python
# Evaluation button
if st.button("Show Evaluate Score", key=f"eval_btn_{len(st.session_state.messages)}"):
    st.session_state.show_eval_panel = True

    # Run evaluation
    with st.spinner("Evaluating response..."):
        try:
            evaluator = RAGASEvaluator(
                st.session_state.rag_system,
                rerank_mode="hybrid"
            )
            from src.evaluation.test_case import EvalCase
            case = EvalCase(
                question=st.session_state.current_query_result.query,
            )
            st.session_state.eval_result = evaluator.run_case(case)
            st.session_state.eval_error = None
        except Exception as e:
            st.session_state.eval_error = f"Evaluation failed: {str(e)}"
            st.session_state.eval_result = None

    st.rerun()
```

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: wire up RAGAS evaluation to button"
```

---

## Task 6: Manual Testing

**Files:**
- None (manual verification)

- [ ] **Step 1: Start Streamlit app**

```bash
cd D:/data/Program/Python/rag-system
python -m streamlit run streamlit_app.py
```

- [ ] **Step 2: Verify button appears**

After getting a response, verify "Show Evaluate Score" button appears below sources.

- [ ] **Step 3: Verify panel behavior**

Click button → sidebar should show spinner → scores should appear when complete.

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: complete evaluate score feature"
```

---

## Verification Checklist

- [ ] Button appears below sources after query response
- [ ] Clicking button opens right sidebar
- [ ] "Evaluating..." spinner shows while running
- [ ] Three scores display: Faithfulness, Answer Relevancy, Context Relevance
- [ ] Close button dismisses panel
- [ ] Error handling works (shows error message in sidebar if eval fails)