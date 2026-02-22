cl /EHsc /O2 /Oi /Ot /arch:AVX2 /fp:fast main.cpp AES_CORE.cpp  /Fe:"AES.exe" | Out-String | %{ 
if ($_ -match "error") { throw $_ } $_ 
} | %{ Write-Host "Compiled Successfully" ;
 .\AES.exe }