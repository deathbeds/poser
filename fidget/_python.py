import nbconvert
import yapf

class YapfPreProcessor(nbconvert.preprocessors.Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell['cell_type'] in ['code']:
            cell['source'] = yapf.yapf_api.FormatCode(cell['source'])[0]
            
        cell['source'] = '\n'.join(
            map(
                lambda s: '' if str.isspace(s) else s.rstrip(), cell['source'].splitlines())
        )

        return cell, resources
    
c.PythonExporter.preprocessors = [YapfPreProcessor]
c.TemplateExporter.template_file = 'output.tpl'