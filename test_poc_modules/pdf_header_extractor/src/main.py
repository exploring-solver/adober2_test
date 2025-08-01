import click
import time
from pathlib import Path
import logging
import json
import os

from src.core.pdf_processor import PDFProcessor
from src.utils.validation import validate_pdf
from config.settings import JSON_OUTPUT_DIR


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file path')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.option('--language', default='auto', help='Document language (auto, en, ja, hi, ar, zh)')
@click.option('--round1a', is_flag=True, help='Use Round 1A hackathon format')
@click.option('--preload', is_flag=True, help='Preload models for faster processing (optional)')
@click.option('--fast-mode', is_flag=True, help='Enable fast mode (skip semantic filtering)')
@click.option('--warmup', is_flag=True, help='Warm up models before processing')
@click.option('--accessibility', is_flag=True, help='Generate accessibility XML output')
@click.option('--metadata', is_flag=True, help='Include full metadata in output (accessibility, document info, etc.)')
def main(pdf_path, output, debug, language, round1a, preload, fast_mode, warmup, accessibility, metadata):
   
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if fast_mode:
        os.environ['FAST_MODE'] = 'true'
        logger.info("Fast mode enabled - semantic filtering disabled")
        
    if metadata:
        os.environ['INCLUDE_METADATA'] = 'true'
        logger.info("Metadata mode enabled - full output with accessibility data")
    else:
        os.environ['INCLUDE_METADATA'] = 'false'
        logger.info("Simple mode enabled - clean output with title and outline only")
    
    startup_time = time.time()
    
    try:        
        logger.info(f"Step: Validating PDF at {pdf_path}")
        if not validate_pdf(pdf_path):
            click.echo(f"Error: Invalid PDF file: {pdf_path}")
            return
        
        logger.info("Step: Initializing PDFProcessor")
        init_start = time.time()
        
        processor = PDFProcessor(language=language, debug=debug)
        
        init_time = time.time() - init_start
        logger.info(f"Processor initialized in {init_time:.3f}s (models not loaded yet)")
        
        logger.info("Step: Preloading models" if preload else "Step: Skipping preload")
        
        if preload:
            logger.info("Preloading models...")
            preload_start = time.time()
            
            if not fast_mode and hasattr(processor, 'semantic_filter') and processor.semantic_filter:
                success = processor.semantic_filter.preload_model()
                if success:
                    logger.info("Semantic model preloaded successfully")
                else:
                    logger.warning("Failed to preload semantic model")
            
            preload_time = time.time() - preload_start
            logger.info(f"Model preloading completed in {preload_time:.3f}s")
        
        logger.info("Step: Warming up models" if warmup else "Step: Skipping warmup")
        
        if warmup:
            logger.info("Warming up models...")
            warmup_start = time.time()
            
            if not fast_mode and hasattr(processor, 'semantic_filter') and processor.semantic_filter:
                processor.semantic_filter.embedding_model.warmup_model([
                    "Introduction", "Chapter 1", "Methodology", 
                    "Results", "Conclusion", "References"
                ])
            
            warmup_time = time.time() - warmup_start
            logger.info(f"Model warmup completed in {warmup_time:.3f}s")
        
        logger.info("Step: Processing PDF")
        logger.info("Starting PDF processing...")
        processing_start = time.time()
        
        if round1a:
            original_metadata = os.getenv('INCLUDE_METADATA')
            os.environ['INCLUDE_METADATA'] = 'false'
            
            result = processor.process(pdf_path)
            
            if original_metadata:
                os.environ['INCLUDE_METADATA'] = original_metadata
            
            result = {
                "title": result.get("title", "Document"),
                "outline": result.get("outline", [])
            }
        else:
            result = processor.process(pdf_path, include_metadata=metadata)
        
        processing_time = time.time() - processing_start
        
        logger.info(f"Step: Saving output to {output}" if output else "Step: Printing output to console")
        if output:
            formats = ["json"]
            if accessibility:
                formats.append("pdf_ua_xml")
            
            processor.save_output(result, output, formats=formats)
            click.echo(f"Results saved to: {output}")
            
            if accessibility:
                accessibility_file = Path(output).with_suffix('_accessibility.xml')
                click.echo(f"Accessibility XML saved to: {accessibility_file}")
        else:
            click.echo(json.dumps(result, indent=4, ensure_ascii=False))
            
            if accessibility and metadata and "accessibility" in result:
                click.echo("\n=== Accessibility Summary ===")
                compliance = result["accessibility"]["compliance_summary"]
                click.echo(f"Accessibility Score: {compliance['accessibility_score']:.1f}/100")
                click.echo(f"WCAG 2.1 AA: {'✓' if compliance['wcag_2_1_aa'] else '✗'}")
                click.echo(f"PDF/UA: {'✓' if compliance['pdf_ua'] else '✗'}")
                click.echo(f"Section 508: {'✓' if compliance['section_508'] else '✗'}")
        
        logger.info("Step: Extraction complete")
        total_time = time.time() - startup_time
        logger.info(f"Processing completed in {processing_time:.2f}s (total: {total_time:.2f}s)")
        
        if debug:
            logger.info("=== Performance Statistics ===")
            logger.info(f"Startup time: {init_time:.3f}s")
            if preload:
                logger.info(f"Preload time: {preload_time:.3f}s")
            if warmup:
                logger.info(f"Warmup time: {warmup_time:.3f}s")
            logger.info(f"Processing time: {processing_time:.3f}s")
            logger.info(f"Total time: {total_time:.3f}s")
            
            if not fast_mode and hasattr(processor, 'semantic_filter') and processor.semantic_filter:
                model_stats = processor.semantic_filter.get_model_info()
                logger.info("=== Model Statistics ===")
                
                if 'cache_stats' in model_stats:
                    cache_stats = model_stats['cache_stats']
                    logger.info(f"Models loaded: {cache_stats.get('models_currently_loaded', 0)}")
                    logger.info(f"Cache hit rate: {cache_stats.get('cache_hit_rate', 'N/A')}")
                
                if 'performance_stats' in model_stats:
                    perf_stats = model_stats['performance_stats']
                    logger.info(f"Total load time: {perf_stats.get('total_load_time', 'N/A')}")
                    logger.info(f"Avg load time: {perf_stats.get('avg_load_time', 'N/A')}")
            
            if isinstance(result, dict):
                outline = result.get('outline', [])
                if outline:
                    logger.info(f"=== Extraction Summary ===")
                    logger.info(f"Total headings: {len(outline)}")
                    
                    level_counts = {}
                    for item in outline:
                        level = item.get('level', 'unknown')
                        level_counts[level] = level_counts.get(level, 0) + 1
                    
                    for level, count in sorted(level_counts.items()):
                        logger.info(f"{level}: {count}")
                
                if accessibility and "accessibility" in result:
                    acc_data = result["accessibility"]["metadata"]
                    logger.info(f"=== Accessibility Statistics ===")
                    logger.info(f"Accessibility Score: {acc_data['accessibility_score']:.1f}/100")
                    logger.info(f"Issues Found: {len(acc_data['issues'])}")
                    for issue in acc_data['issues']:
                        logger.info(f"  - {issue}")
        
        if debug:
            logger.info("Tip: Models remain in memory for faster subsequent processing")
            logger.info("Use processor.clear_caches() to free memory if needed")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        if debug:
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        click.echo(f"Error: {str(e)}")
        return 1

@click.command()
@click.option('--model-info', is_flag=True, help='Show model information')
@click.option('--clear-cache', is_flag=True, help='Clear all model caches')
@click.option('--benchmark', is_flag=True, help='Run performance benchmark')
@click.option('--accessibility-check', is_flag=True, help='Check accessibility compliance')
def utils(model_info, clear_cache, benchmark, accessibility_check):
    
    if model_info:
        from src.models.embedding_model import EmbeddingModel
        
        model = EmbeddingModel()
        info = model.get_model_info()
        
        click.echo("=== Model Information ===")
        click.echo(json.dumps(info, indent=2))
    
    if clear_cache:
        from src.models.embedding_model import EmbeddingModel
        
        model = EmbeddingModel()
        model.clear_cache()
        click.echo("All model caches cleared")
    
    if benchmark:
        from src.models.embedding_model import EmbeddingModel
        import time
        
        click.echo("Running performance benchmark...")
        
        model = EmbeddingModel()
        
        start = time.time()
        embedding1 = model.encode("Test text for benchmark")
        first_time = time.time() - start
        
        start = time.time()
        embedding2 = model.encode("Another test text")
        second_time = time.time() - start
        
        test_texts = [f"Heading {i}" for i in range(10)]
        start = time.time()
        batch_embeddings = model.encode(test_texts)
        batch_time = time.time() - start
        
        click.echo(f"First embedding (with loading): {first_time:.3f}s")
        click.echo(f"Second embedding (cached): {second_time:.3f}s")
        click.echo(f"Batch processing (10 texts): {batch_time:.3f}s")
        click.echo(f"Speedup factor: {first_time/second_time:.1f}x")
        
        stats = model.get_performance_stats()
        click.echo("\nPerformance Statistics:")
        click.echo(json.dumps(stats, indent=2))
    
    if accessibility_check:
        from src.core.accessibility_tagger import AccessibilityTagger
        
        sample_headings = [
            {"text": "Introduction", "level": "H1", "page": 1},
            {"text": "Background", "level": "H2", "page": 1},
            {"text": "Methodology", "level": "H1", "page": 2},
            {"text": "Data Collection", "level": "H2", "page": 2},
            {"text": "Analysis", "level": "H2", "page": 3},
            {"text": "Results", "level": "H1", "page": 4},
            {"text": "Conclusion", "level": "H1", "page": 5}
        ]
        
        tagger = AccessibilityTagger()
        accessibility_data = tagger.generate_accessibility_metadata(sample_headings)
        
        click.echo("=== Accessibility Compliance Check ===")
        click.echo(f"Accessibility Score: {accessibility_data['accessibility_score']:.1f}/100")
        click.echo(f"WCAG 2.1 AA: {'✓' if accessibility_data['compliance']['wcag_2.1_aa'] else '✗'}")
        click.echo(f"PDF/UA: {'✓' if accessibility_data['compliance']['pdf_ua'] else '✗'}")
        click.echo(f"Section 508: {'✓' if accessibility_data['compliance']['section_508'] else '✗'}")
        
        if accessibility_data['issues']:
            click.echo("\nIssues Found:")
            for issue in accessibility_data['issues']:
                click.echo(f"  - {issue}")
        
        click.echo("\nRecommendations:")
        for rec in accessibility_data['recommendations']:
            click.echo(f"  - {rec}")


@click.group()
def cli():
    pass

cli.add_command(main, name='extract')
cli.add_command(utils, name='utils')

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['extract', 'utils']:
        cli()
    else:
        main()