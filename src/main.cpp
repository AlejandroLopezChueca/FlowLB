#include "ui/app.h"

int main()
{
  if (FLB::App::s_AppCreated) return 0;
  FLB::App app;
  app.startApp();
  return 0;
}
