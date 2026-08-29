# Tarjeta de modelo — riesgo de revisión

Este modelo predice **riesgo de revisión manual** como etiqueta binaria (0/1):
si una factura debería pasar por un analista antes de pagarse.

**No predice fraude real.** Un 1 no significa que haya delito; solo que hay
señales de inconsistencia o rareza que justifican revisión humana.

**Features:** `amount`, `lines_sum`, `hour`, `is_new_vendor`,
`vendor_avg_amount`, `amount_vs_avg_ratio`. No se usan identificadores
(`invoice_id`, `vendor_id`, `vendor_name`).

La etiqueta `risk` se **bootstrappea con reglas** (ratio vs. promedio del
proveedor, descuadre amount/lines_sum, proveedor nuevo, hora nocturna) y
luego se **invierte en un 8%** de filas (ruido de etiqueta).

**Límites conocidos:** los datos son sintéticos; el rendimiento no se
generaliza a facturas reales. Si `vendor_avg_amount` se calcula mal en
producción (p. ej. incluyendo la factura actual o usando otro universo de
proveedores) hay **riesgo de leakage / drift** y las predicciones dejan de
ser comparables al entrenamiento.
