import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better looking plots
sns.set_style("whitegrid")

# Function to read different file types
def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

# Function to calculate similarity
def calculate_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    except ValueError:
        # Handle cases where vectorization fails (e.g., only numbers or special chars)
        return 0.0

# Function to compare multiple files
def compare_multiple_files(files):
    results = []
    similarity_matrix = []
    texts = []
    
    # Read all files first
    for file in files:
        try:
            texts.append(read_file(file))
        except Exception as e:
            st.error(f"Error reading {file.name}: {str(e)}")
            texts.append("")
    
    # Compare each pair
    for i in range(len(files)):
        row = []
        numeric_row = []
        for j in range(len(files)):
            if i == j:
                row.append("--")
                numeric_row.append(100.0)  # 100% similarity with itself
            elif i < j:
                similarity = calculate_similarity(texts[i], texts[j])
                row.append(f"{similarity:.2f}%")
                numeric_row.append(similarity)
            else:
                # Use previously calculated symmetric value
                row.append(results[j][i])
                numeric_row.append(similarity_matrix[j][i])
        results.append(row)
        similarity_matrix.append(numeric_row)
    
    return results, [file.name for file in files], similarity_matrix

def plot_similarity_histogram(similarity_matrix, filenames):
    """Plot histogram of similarity scores."""
    # Flatten the matrix and remove self-comparisons (100% values)
    flat_scores = []
    for i in range(len(similarity_matrix)):
        for j in range(i+1, len(similarity_matrix)):
            flat_scores.append(similarity_matrix[i][j])
    
    if not flat_scores:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(flat_scores, bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Similarity Scores')
    ax.set_xlabel('Similarity Score (%)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 100)
    
    # Add mean and median lines
    mean_score = np.mean(flat_scores)
    median_score = np.median(flat_scores)
    
    ax.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.1f}%')
    ax.axvline(median_score, color='g', linestyle='-.', label=f'Median: {median_score:.1f}%')
    ax.legend()
    
    return fig

def plot_similarity_heatmap(similarity_matrix, filenames):
    """Plot heatmap of similarity scores."""
    # Convert to numpy array for plotting
    matrix_array = np.array(similarity_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.zeros_like(matrix_array)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Plot heatmap
    sns.heatmap(matrix_array, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=filenames, yticklabels=filenames,
                mask=mask, vmin=0, vmax=100, ax=ax,
                cbar_kws={'label': 'Similarity Score (%)'})
    
    ax.set_title('Document Similarity Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return fig

# Streamlit UI
st.title("Plagiarism Checker")

mode = st.radio("Select comparison mode:", 
                ["Pairwise (2 files/texts)", "Multiple files (compare all)"], 
                horizontal=True)

if mode == "Pairwise (2 files/texts)":
    st.write("### Option 1: Upload two files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("First file", type=["txt", "pdf", "docx"], key="file1")
    with col2:
        file2 = st.file_uploader("Second file", type=["txt", "pdf", "docx"], key="file2")

    st.write("### Option 2: Or paste texts directly")
    text1 = st.text_area("First text", height=150, key="text1")
    text2 = st.text_area("Second text", height=150, key="text2")

    if st.button("Compare"):
        # Determine source of input
        if file1 is not None and file2 is not None:
            try:
                text1 = read_file(file1)
                text2 = read_file(file2)
            except Exception as e:
                st.error(f"Error reading files: {e}")
                st.stop()
        elif not text1.strip() or not text2.strip():
            st.warning("Please upload both files or paste both texts.")
            st.stop()

        similarity = calculate_similarity(text1, text2)
        st.success(f"Similarity Score: {similarity:.2f}%")
        
        # Plot histogram for pairwise comparison (though not very meaningful)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Similarity'], [similarity], color='skyblue')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Similarity (%)')
        ax.set_title('Pairwise Similarity Score')
        st.pyplot(fig)
        
        # Show text previews
        with st.expander("View Text Comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Text 1")
                st.text(text1[:2000] + ("..." if len(text1) > 2000 else ""))
            with col2:
                st.write("### Text 2")
                st.text(text2[:2000] + ("..." if len(text2) > 2000 else ""))

else:  # Multiple files mode
    st.write("### Upload multiple files for comparison")
    uploaded_files = st.file_uploader("Choose files", 
                                    type=["txt", "pdf", "docx"], 
                                    accept_multiple_files=True)
    
    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 files for comparison.")
    elif st.button("Compare All Files"):
        with st.spinner("Analyzing files..."):
            results, filenames, similarity_matrix = compare_multiple_files(uploaded_files)
        
        st.success("Comparison complete!")
        
        # Display results as a matrix
        st.write("### Similarity Matrix")
        
        # Create a table header
        header = "| File | " + " | ".join(filenames) + " |"
        separator = "|-----|" + "|".join([":---:" for _ in filenames]) + "|"
        
        # Build the table rows
        rows = []
        for i, row in enumerate(results):
            row_str = f"| {filenames[i]} | " + " | ".join(row) + " |"
            rows.append(row_str)
        
        # Display the markdown table
        st.markdown("\n".join([header, separator] + rows))
        
        # Visualization section
        st.write("## Data Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Heatmap", "Histogram"])
        
        with tab1:
            st.write("### Similarity Heatmap")
            heatmap_fig = plot_similarity_heatmap(similarity_matrix, filenames)
            st.pyplot(heatmap_fig)
            st.write("""
            **Heatmap Interpretation:**
            - Darker colors indicate higher similarity between documents
            - The diagonal is intentionally masked as it represents self-comparison (100%)
            - Upper triangle is shown to avoid duplicate comparisons
            """)
        
        with tab2:
            st.write("### Similarity Distribution Histogram")
            hist_fig = plot_similarity_histogram(similarity_matrix, filenames)
            if hist_fig:
                st.pyplot(hist_fig)
                st.write("""
                **Histogram Interpretation:**
                - Shows the distribution of all pairwise similarity scores
                - Red dashed line indicates the mean similarity
                - Green dash-dot line indicates the median similarity
                """)
            else:
                st.warning("Not enough data to generate histogram")
        
        # Add download option
        csv_content = "File," + ",".join(filenames) + "\n"
        for i, row in enumerate(results):
            csv_content += filenames[i] + "," + ",".join(row) + "\n"
        
        st.download_button(
            label="Download Results as CSV",
            data=csv_content,
            file_name="plagiarism_results.csv",
            mime="text/csv"
        )

# Add some instructions
st.markdown("""
### Instructions:
1. For pairwise comparison:
   - Upload two files OR paste two texts
   - Click "Compare" button
   
2. For multiple files comparison:
   - Upload 2 or more files
   - Click "Compare All Files" button
   - View the similarity matrix and visualizations

**Visualizations Include:**
- Heatmap showing document similarity patterns
- Histogram showing distribution of similarity scores

Note: The similarity score is calculated using TF-IDF and cosine similarity.
""")