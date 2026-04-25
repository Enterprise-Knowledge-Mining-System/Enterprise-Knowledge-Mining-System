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


def preview_section_blocks(blocks, text_limit=500):
    for block in blocks:
        metadata = block.get("metadata", {})
        print(f"\nPAGE: {metadata.get('page_number')}")
        print(f"TITLE: {metadata.get('title')}")
        print(f"SECTION: {block['section']}")
        print(f"TEXT SAMPLE: {block['text'][:text_limit]}")

    print(f"\nTotal sections extracted: {len(blocks)}")


def display_search_results(results, preview_chars=300):
    if not results:
        print("No results found.")
        return

    for result in results:
        print("=" * 80)
        print(f"Rank: {result['rank']}")
        print(f"Paper ID: {result['paper_id']}")
        print(f"Title: {result['title']}")
        print(f"Section: {result['section']}")
        print(f"Primary Category: {result['primary_category']}")
        print(f"Categories: {result['categories']}")
        print(f"Authors: {result['authors']}")
        print(f"Published: {result['published']}")
        print(
            f"Page: {result['page_number']} | "
            f"Block: {result['page_block_index']} | "
            f"Page Chunk: {result['page_chunk_index']} | "
            f"Global Chunk: {result['chunk_index']}"
        )
        print(f"Cosine Similarity (approx): {result['cosine_similarity']}")
        print(f"Entities: {result['entities']}")
        print(f"Entity Labels: {result['entity_labels']}")
        print(f"Preview: {result['text'][:preview_chars]}")
        print()
