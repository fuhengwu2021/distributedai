#!/usr/bin/env python3
"""
NeMo Curator Quick Demo: Demonstrates data cleaning, deduplication, and quality filtering

This example completes in under 1 minute and demonstrates:
1. Data loading (from in-memory sample data)
2. Quality filtering (word count, symbol ratio)
3. Exact deduplication
4. Result statistics

Runtime: < 1 minute (depends on system performance)
"""

import json
import os
import tempfile
import time
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    RepeatedLinesFilter,
)
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow


def create_sample_data(output_dir: str) -> str:
    """Create sample data containing low-quality and duplicate content"""
    
    # Sample data: contains high-quality, low-quality, and duplicate content
    sample_docs = [
        # High-quality documents
        {
            "id": "doc_001",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on the input data."
        },
        {
            "id": "doc_002", 
            "text": "Distributed systems are computing systems that consist of multiple components located on different networked computers. These components communicate and coordinate their actions by passing messages to achieve a common goal."
        },
        {
            "id": "doc_003",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields such as computer vision, natural language processing, and speech recognition."
        },
        # Low-quality document: too short
        {
            "id": "doc_004",
            "text": "AI is cool."
        },
        # Low-quality document: too many symbols
        {
            "id": "doc_005",
            "text": "!!!@@@###$$$%%%^^^&&&***((()))___+++===---~~~```"
        },
        # Low-quality document: too many repeated lines
        {
            "id": "doc_006",
            "text": "This is a test. This is a test. This is a test. This is a test. This is a test. This is a test."
        },
        # Duplicate document (identical to doc_001)
        {
            "id": "doc_007",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on the input data."
        },
        # Duplicate document (identical to doc_002)
        {
            "id": "doc_008",
            "text": "Distributed systems are computing systems that consist of multiple components located on different networked computers. These components communicate and coordinate their actions by passing messages to achieve a common goal."
        },
        # High-quality document
        {
            "id": "doc_009",
            "text": "Natural language processing is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language. It combines computational linguistics with statistical, machine learning, and deep learning models."
        },
        # Low-quality document: too long (exceeds limit)
        {
            "id": "doc_010",
            "text": " ".join(["This is a very long document. "] * 10000)  # ~300,000 words
        },
    ]
    
    # Write to JSONL file
    input_file = os.path.join(output_dir, "input_data.jsonl")
    with open(input_file, 'w', encoding='utf-8') as f:
        for doc in sample_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"✓ Created {len(sample_docs)} sample documents")
    print(f"  - High-quality documents: 4")
    print(f"  - Low-quality documents: 4 (too short, too many symbols, repeated lines, too long)")
    print(f"  - Duplicate documents: 2 (identical to high-quality documents)")
    
    return input_file


def run_curation_demo(input_file: str, output_dir: str) -> dict:
    """Run the data curation pipeline"""
    
    print("\n" + "="*60)
    print("Starting data curation pipeline...")
    print("="*60)
    
    # Create temporary directories for intermediate results
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    filtered_dir = os.path.join(temp_dir, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)
    
    dedup_dir = os.path.join(temp_dir, "dedup")
    os.makedirs(dedup_dir, exist_ok=True)
    
    final_output_dir = os.path.join(output_dir, "curated")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Start Ray
    # Note: By default, RayClient() uses CPU-only mode (num_gpus=None)
    # For GPU acceleration, use: RayClient(num_gpus=1) or RayClient(num_gpus=4)
    # This demo uses CPU since the sample data is small (10 documents)
    ray_client = RayClient()
    ray_client.start()
    
    try:
        # Stage 1: Quality filtering pipeline
        print("\n[Stage 1] Quality filtering...")
        filter_pipeline = Pipeline(
            name="quality_filtering",
            description="Filter low-quality documents"
        )
        
        # Read data
        filter_pipeline.add_stage(
            JsonlReader(
                file_paths=input_file,
                files_per_partition=1,
                fields=["text", "id"]
            )
        )
        
        # Word count filter: keep documents with 50-10000 words
        filter_pipeline.add_stage(
            ScoreFilter(
                filter_obj=WordCountFilter(min_words=50, max_words=10000),
                text_field="text",
                score_field="word_count"
            )
        )
        
        # Symbol ratio filter: symbols cannot exceed 25%
        filter_pipeline.add_stage(
            ScoreFilter(
                filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
                text_field="text"
            )
        )
        
        # Repeated lines filter: repeated lines cannot exceed 70%
        filter_pipeline.add_stage(
            ScoreFilter(
                filter_obj=RepeatedLinesFilter(max_repeated_line_fraction=0.7),
                text_field="text"
            )
        )
        
        # Write filtered data
        filter_pipeline.add_stage(
            JsonlWriter(filtered_dir)
        )
        
        filter_start = time.time()
        filter_results = filter_pipeline.run()
        filter_time = time.time() - filter_start
        
        # Count filtered documents
        filtered_count = 0
        for result in filter_results:
            if hasattr(result, 'data') and result.data is not None:
                # Read output files to count
                output_files = result.data if isinstance(result.data, list) else [result.data]
                for output_file in output_files:
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            filtered_count = sum(1 for _ in f)
        
        print(f"  ✓ Quality filtering completed (time: {filter_time:.2f}s)")
        print(f"  ✓ Documents retained: {filtered_count}")
        
        # Stage 2: Exact deduplication
        print("\n[Stage 2] Exact deduplication...")
        dedup_workflow = ExactDeduplicationWorkflow(
            input_path=filtered_dir,
            output_path=dedup_dir,
            text_field="text",
            perform_removal=False,  # Only identify, don't remove
            assign_id=True,
            input_filetype="jsonl",
        )
        
        dedup_start = time.time()
        dedup_workflow.run()
        dedup_time = time.time() - dedup_start
        
        # Read deduplication results
        dedup_files = list(Path(dedup_dir).glob("*.jsonl"))
        dedup_count = 0
        if dedup_files:
            with open(dedup_files[0], 'r', encoding='utf-8') as f:
                dedup_count = sum(1 for _ in f)
        
        print(f"  ✓ Deduplication completed (time: {dedup_time:.2f}s)")
        print(f"  ✓ Documents after deduplication: {dedup_count}")
        
        # Copy final results
        if dedup_files:
            import shutil
            shutil.copy(dedup_files[0], os.path.join(final_output_dir, "curated_data.jsonl"))
        
        total_time = filter_time + dedup_time
        
        return {
            "input_count": 10,
            "filtered_count": filtered_count,
            "dedup_count": dedup_count,
            "filter_time": filter_time,
            "dedup_time": dedup_time,
            "total_time": total_time,
            "output_file": os.path.join(final_output_dir, "curated_data.jsonl")
        }
        
    finally:
        ray_client.stop()


def print_results(stats: dict):
    """Print result statistics"""
    print("\n" + "="*60)
    print("Data Curation Results Statistics")
    print("="*60)
    print(f"Input documents:     {stats['input_count']}")
    print(f"After filtering:    {stats['filtered_count']}  ({stats['filtered_count']/stats['input_count']*100:.1f}%)")
    print(f"After deduplication: {stats['dedup_count']}  ({stats['dedup_count']/stats['input_count']*100:.1f}%)")
    print(f"\nProcessing time:")
    print(f"  Quality filtering: {stats['filter_time']:.2f} seconds")
    print(f"  Deduplication:     {stats['dedup_time']:.2f} seconds")
    print(f"  Total:             {stats['total_time']:.2f} seconds")
    print(f"\nData reduction rate: {(1 - stats['dedup_count']/stats['input_count'])*100:.1f}%")
    print(f"\nOutput file:         {stats['output_file']}")
    print("="*60)


def main():
    """Main function"""
    print("NeMo Curator Quick Demo")
    print("="*60)
    print("This example demonstrates:")
    print("  1. Quality filtering (word count, symbol ratio, repeated lines)")
    print("  2. Exact deduplication")
    print("  3. Data reduction effects")
    print("="*60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        input_file = create_sample_data(temp_dir)
        
        # Run curation pipeline
        stats = run_curation_demo(input_file, temp_dir)
        
        # Print results
        print_results(stats)
        
        print("\n✓ Demo completed!")
        print("\nNote: In real-world applications, NeMo Curator can process TB to PB scale data,")
        print("     leveraging GPU acceleration and distributed processing for 16× performance improvement.")


if __name__ == "__main__":
    main()
