from pathlib import Path
import sys, xml.etree.ElementTree as ET, zipfile
version=sys.argv[1] if len(sys.argv)>1 else ET.parse('Directory.Build.props').findtext('./PropertyGroup/Version')
for project in sorted(Path('src').glob('*/*.csproj')):
    name=project.stem
    package=Path(f'.out/packages/{name}.{version}.nupkg')
    assert package.exists() and package.with_suffix('.snupkg').exists()
    with zipfile.ZipFile(package) as archive:
        manifest=ET.fromstring(archive.read(name+'.nuspec'));ns={'n':manifest.tag.split('}')[0][1:]}
        dependencies={d.attrib['id']:d.attrib['version'] for d in manifest.findall('.//n:dependency',ns)}
        assert f'lib/net10.0/{name}.dll' in archive.namelist()
        assert manifest.find('n:metadata/n:version',ns).text==version
        source=ET.parse(project)
        expected={Path(d.attrib['Include']).stem:f'[{version}]' for d in source.findall('.//ProjectReference')}
        central={d.attrib['Include']:d.attrib['Version'] for d in ET.parse('Directory.Packages.props').findall('.//PackageVersion')}
        expected.update({d.attrib['Include']:central[d.attrib['Include']] for d in source.findall('.//PackageReference')})
        assert dependencies==expected,(name,dependencies,expected)
        assert not any(n in dependencies for n in ['FScript.Language','MagnusOpera.FScript.Runtime','FScript.Runtime'])
        if name!='FScript.Numerics.Extensions':assert not any('FScript.Extensibility' in n or n=='FSharp.Core' for n in dependencies)
    print(f'{name}: package, symbols, and dependency graph verified')
