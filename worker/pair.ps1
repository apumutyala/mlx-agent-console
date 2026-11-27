# Shows the QR code that pairs the iPad app with this worker. See pair.py.
param([switch]$Ascii)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$extra = @()
if ($Ascii) { $extra += "--ascii" }
uv run --no-project --with "qrcode[pil]" python "$here\pair.py" @extra
