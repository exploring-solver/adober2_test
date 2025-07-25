import click
import time
from pathlib import Path
from core.pdf_processor import PDFProcessor
from utils.validation import validate_pdf
import logging

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output JSON file path')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--language', default='auto', help='Document language (auto, en, ja, hi, ar, zh)')
def main(pdf_path, output, debug, language):
    """Extract headings from PDF using hybrid approach."""
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # Validate PDF
        if not validate_pdf(pdf_path):
            click.echo(f"Error: Invalid PDF file: {pdf_path}")
            return
        
        # Initialize processor
        processor = PDFProcessor(language=language, debug=debug)
        
        # Process PDF
        result = processor.process(pdf_path)
        
        # Save output
        if output:
            processor.save_output(result, output)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        click.echo(f"Error: {str(e)}")

if __name__ == '__main__':
    main()