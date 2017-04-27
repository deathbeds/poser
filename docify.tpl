{%- extends 'null.tpl' -%}
{% block header %}# coding: utf-8

{% endblock header %}


{% macro docify(source, doc) %}
{% filter ipython2python | trim %}
{% if ':)'[::-1] in source %}
{% set signature, rest = source.split('):', 1) %}
{{signature}}):{% if doc.strip() %}
{% set spaces = 4 + signature.splitlines()[-1].__len__() - signature.splitlines()[-1].lstrip().__len__() %}
{% filter wrap_text(80-spaces) | trim | indent(spaces) %}"""{{ doc }}
"""
{% endfilter %}{% endif %}{{rest}}
{% else -%}
{% if doc.strip() %}{{doc | trim | comment_lines}}

{% endif %}{{source}}
{%- endif %}
{% endfilter %}
{% endmacro %}

{%- block body -%}
{%- for cell in nb.cells -%}
{%- if 'markdown' in cell.cell_type and loop.index < loop.length and 'code' in nb.cells[loop.index].cell_type -%}
{%- elif 'code' in cell.cell_type %}{% set prev = nb.cells[loop.index0-1] -%}{{docify(
        cell.source, prev.source if loop.index and 'markdown' in prev.cell_type else ""
    )}}
{%- elif 'markdown' in cell.cell_type -%}{{ cell.source | trim  | comment_lines }}
{%- endif -%}

{%- endfor -%}
{%- endblock body -%}
