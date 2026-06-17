"""
Definición de tipos comunes para el Statistical Toolbelt.

Este módulo contiene el tipo de retorno central ToolbeltResult que permite
la adaptación automática al entorno de ejecución (Jupyter vs script).
"""

import base64
import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from matplotlib.figure import Figure


@dataclass
class ToolbeltResult:
    """Resultado de una función de diagnóstico con datos y visualización.
    
    Este tipo es el mecanismo central que permite la adaptación automática
    al entorno de ejecución. En Jupyter, el método _repr_html_ es llamado
    automáticamente para renderizar la tabla como HTML con la figura inline.
    En consola, __str__ genera una representación de texto plano.
    
    Attributes:
        data: DataFrame con los resultados numéricos del diagnóstico.
        figure: Figura de matplotlib asociada al diagnóstico (puede ser None).
        title: Título descriptivo del diagnóstico.
    """
    
    data: pd.DataFrame
    figure: Optional[Figure] = None
    title: str = ""
    
    def _repr_html_(self) -> str:
        """Renderizado automático en Jupyter.
        
        Jupyter llama este método automáticamente al evaluar la variable.
        Genera HTML con el título, la tabla con scroll y la figura como
        imagen base64 inline.
        """
        # Generar tabla HTML con scroll
        table_html = self.data.to_html(
            classes="toolbelt-table",
            index=False,
            border=0,
            max_rows=None,
            max_cols=None
        )
        
        # Estilo CSS para la tabla
        table_style = """
        <style>
            .toolbelt-table {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                font-size: 12px;
                border-collapse: collapse;
                width: 100%;
                max-height: 300px;
                overflow-y: auto;
                display: block;
            }
            .toolbelt-table th {
                background-color: #f8f9fa;
                padding: 8px;
                text-align: left;
                border-bottom: 2px solid #dee2e6;
                position: sticky;
                top: 0;
            }
            .toolbelt-table td {
                padding: 6px 8px;
                border-bottom: 1px solid #dee2e6;
            }
            .toolbelt-table tr:hover {
                background-color: #f8f9fa;
            }
        </style>
        """
        
        # Generar figura como imagen base64 si existe
        figure_html = ""
        if self.figure is not None:
            buf = io.BytesIO()
            try:
                self.figure.savefig(
                    buf,
                    format="png",
                    dpi=80,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none"
                )
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                figure_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; margin-top:10px;"/>'
            except Exception:
                figure_html = "<p><em>Error al generar la figura</em></p>"
        
        # Construir HTML completo
        title_html = f"<h3>{self.title}</h3>" if self.title else ""
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
            {title_html}
            {table_style}
            {table_html}
            {figure_html}
        </div>
        """
    
    def __str__(self) -> str:
        """Representación en texto plano para consola.
        
        El método print() en consola invoca esta representación,
        que muestra el título seguido de la tabla formateada como texto.
        """
        lines = []
        if self.title:
            lines.append(f"=== {self.title} ===")
        lines.append(self.data.to_string())
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representación técnica que usa __str__."""
        return self.__str__()