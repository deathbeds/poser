{%- extends 'null.tpl' -%}
{% set doc = "" %}

{% block header %}# coding: utf-8

{% endblock header %}

{% macro docify(source, doc=doc) %}
{% set signature, rest = source.split('):', 1) %}
{{signature}}):
{% if doc.strip() %}    """{{ doc | wrap_text(80) | indent(4, False) | trim }}
    """{% endif -%}
{{rest}}
{% endmacro %}



{%- block body -%}
    {%- for cell in nb.cells -%}
        {%- if cell.cell_type == 'code' -%}
            {% if '):' in cell.source and 'class ' in cell.source or 'def ' in cell.source -%}
{% set source = docify(cell.source, doc) %}
{{ source | ipython2python}}
            {%- else %}
{% if doc.strip() -%}{{ doc | comment_lines }}
{% endif %}
{{ cell.source | ipython2python}}
            {%- endif -%}
            {% set doc = "" %}
        {%- elif cell.cell_type in ['markdown'] -%}
            {% set doc = cell.source %}
        {%- endif -%}
    {%- endfor -%}
{%- endblock body -%}
