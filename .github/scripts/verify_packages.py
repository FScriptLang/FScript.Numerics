from pathlib import Path
import sys, xml.etree.ElementTree as ET, zipfile
version = sys.argv[1] if len(sys.argv)>1 else ET.parse('Directory.Build.props').findtext('./PropertyGroup/Version')
namespace = {'n':'http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd'}
for name in ['FScript.Numerics','FScript.Numerics.Extensions']:
    package = Path(f'.out/packages/{name}.{version}.nupkg')
    assert package.exists() and package.with_suffix('.snupkg').exists()
    with zipfile.ZipFile(package) as archive:
        manifest = ET.fromstring(archive.read(name + '.nuspec'))
        ns = {'n': manifest.tag.split('}')[0][1:]}
        dependencies = {d.attrib['id']:d.attrib['version'] for d in manifest.findall('.//n:dependency',ns)}
        assert f'lib/net10.0/{name}.dll' in archive.namelist()
        assert manifest.find('n:metadata/n:version',ns).text == version
        if name == 'FScript.Numerics':
            assert dependencies == {'System.Numerics.Tensors':'10.0.12'}, dependencies
        else:
            assert dependencies == {'FScript.Numerics':f'[{version}]','MagnusOpera.FScript.Extensibility':'1.0.0'}, dependencies
    print(f'{name}: package, symbols, and dependency graph verified')
