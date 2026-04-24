# ==== Printing API Response Helper Function ====
def print_papers(papers, summary_len=200):
    for paper in papers:
        print("ARXIV ID:", paper["arxiv_id"])
        print("\nTITLE:", paper["title"])
        print("AUTHORS:", ", ".join(paper["authors"]))
        print("PUBLISHED:", paper["published"])
        print("SUMMARY:", paper["summary"][:summary_len], "...")
        print("PRIMARY CATEGORY:", paper["primary_category"])
        print("CATEGORIES:", ", ".join(paper["categories"]))
        print("PDF LINK:", paper["pdf_link"])
        print("\n" + "-" * 50 + "\n")


# ==== Page Inspection Helper Functions ====
def get_page_metadata(pages, index=0):
    return pages[index]["metadata"]


def get_page_text(pages, index=0, limit=500):
    text = pages[index]["text"]
    return text[:limit] if limit else text


def preview_page(pages, index=0, char_limit=500):
    print(f"\n--- Page {index} Preview ---")
    print("Metadata:", get_page_metadata(pages, index))
    print("Text Preview:\n", get_page_text(pages, index, char_limit))


# ==== Cleaned Paper Inspection Helper Function ====
def preview_cleaned_paper(cleaned_papers, paper_index=0, page_index=0, char_limit=500):
    if not cleaned_papers:
        print("No cleaned papers available.")
        return

    paper = cleaned_papers[paper_index]

    print(f"\n==== {paper['title']} ====")
    preview_page(paper["pages"], index=page_index, char_limit=char_limit)


# ==== Debugging Helper Functions ====

#checking for short/empty pages that might cause issues with NER
def print_short_pages(processed_papers, min_length=50):
    for paper in processed_papers:
        for i, page in enumerate(paper["pages"]):
            if len(page["text"].strip()) < min_length:
                print(f"Short/empty page found in '{paper['title']}' (Page {i}): {len(page['text'])} characters")


def print_page_length_stats(processed_papers):
    lengths = []

    for paper in processed_papers:
        lengths.extend([len(page["text"]) for page in paper["pages"]])

    if not lengths:
        return None
    
    print(f"Total pages: {len(lengths)}")
    print(f'Average chars per page: {sum(lengths)/len(lengths):.0f}')
    print(f"Min: {min(lengths)}, Max: {max(lengths)}")