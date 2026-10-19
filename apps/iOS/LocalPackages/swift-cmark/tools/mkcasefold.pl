binmode STDOUT;
print("    switch (c) {\n");
my $lastchar = "";
while (<STDIN>) {
  if (/^[A-F0-9]/ and / [CF]; /) {
    my ($char, $type, $subst) = m/([A-F0-9]+); ([CF]); ([^;]+)/;
    if ($char eq $lastchar) {
      break;
    }
    my @subst = $subst =~ m/(\w+)/g;
    printf("      case 0x%s:\n", $char);
    foreach (@subst) {
      printf("        bufpush(0x%s);\n", $_);
    }
    printf("        break;\n");
    $lastchar = $char;
  }
}
printf("      default:\n");
printf("        bufpush(c);\n");
print("    }\n");

