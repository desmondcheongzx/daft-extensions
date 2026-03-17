# daft-extensions

Native extensions for the [Daft](https://daft.ai) data engine, built on Daft's [extension system](https://docs.daft.ai/en/stable/extensions/).

## Extensions

| Package | Description | Install |
|---|---|---|
| [daft-h3](h3/) | [H3](https://h3geo.org/) geospatial indexing functions | `pip install daft-h3` |

## Usage

Extensions are loaded into a Daft session at runtime:

```python
import daft
import daft_h3
from daft.session import Session

sess = Session()
sess.load_extension(daft_h3)

with sess:
    df = daft.from_pydict({"lat": [37.7749], "lng": [-122.4194]})
    df = df.select(daft_h3.h3_latlng_to_cell(daft.col("lat"), daft.col("lng"), 7).alias("cell"))
    df.collect().show()
```

See each extension's README for full documentation.
