#pragma once
#include "samplerz.hpp"

namespace test {

// Test correctness of samplerz implementation using test vectors provided with
// in table 3.2 of Falcon specification https://falcon-sign.info/falcon.pdf
void
samplerz()
{
  // see row 1 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -91.90471153063714;
    const double sigma = 1.7037990414754918;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 15,  197, 68, 47,  240, 67, 214, 110, 145, 209, 234,
                          202, 198, 78, 165, 69,  10, 34,  148, 30,  220, 108 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -92);
  }

  // see row 2 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -8.322564895434937;
    const double sigma = 1.7037990414754918;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = {
      244, 218, 15,  141, 132, 68, 209, 167, 114, 101, 194,
      239, 111, 152, 187, 187, 75, 238, 125, 184, 217, 179
    };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -8);
  }

  // see row 3 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -19.096516109216804;
    const double sigma = 1.7035823083824078;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 219, 71, 246, 215, 251, 155, 25,  242, 92,  54, 214,
                          185, 51, 77,  71,  122, 139, 192, 190, 104, 20, 93 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -20);
  }

  // see row 4 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -11.335543982423326;
    const double sigma = 1.7035823083824078;
    const double sigmin = 1.277833697;

    uint8_t bytes[44] = {
      174, 65,  180, 245, 32, 150, 101, 199, 77,  0,   220, 193, 168, 22,  138,
      123, 181, 22,  179, 25, 12,  180, 44,  29,  237, 38,  205, 82,  174, 215,
      112, 236, 167, 221, 51, 78,  5,   71,  188, 195, 193, 99,  206, 11
    };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -12);
  }

  // see row 5 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = 7.9386734193997555;
    const double sigma = 1.6984647769450156;
    const double sigmin = 1.277833697;

    uint8_t bytes[33] = {
      49,  5,   65,  102, 193, 1,   39,  128, 198, 3,   174,
      155, 131, 60,  236, 115, 242, 244, 28,  165, 128, 124,
      200, 156, 146, 21,  136, 52,  99,  47,  155, 21,  85
    };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == 8);
  }

  // see row 6 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -28.990850086867255;
    const double sigma = 1.6984647769450156;
    const double sigmin = 1.277833697;

    uint8_t bytes[11] = { 115, 126, 157, 104, 165, 10, 6, 219, 188, 100, 119 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -30);
  }

  // see row 7 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -9.071257914091655;
    const double sigma = 1.6980782114808988;
    const double sigmin = 1.277833697;

    uint8_t bytes[11] = { 169, 141, 221, 20, 191, 11, 242, 32, 97, 214, 50 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -10);
  }

  // see row 8 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -43.88754568839566;
    const double sigma = 1.6980782114808988;
    const double sigmin = 1.277833697;

    uint8_t bytes[11] = { 60, 191, 104, 24, 166, 143, 122, 185, 153, 21, 20 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -41);
  }

  // see row 9 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -58.17435547946095;
    const double sigma = 1.7010983419195522;
    const double sigmin = 1.277833697;

    uint8_t bytes[23] = { 111, 134, 51,  245, 191, 165, 210, 104,
                          72,  102, 142, 61,  93,  221, 70,  149,
                          142, 151, 99,  4,   16,  88,  124 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -61);
  }

  // see row 10 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -43.58664906684732;
    const double sigma = 1.7010983419195522;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 39, 43, 198, 194, 95,  92,  94, 229, 63,  131, 196,
                          58, 54, 31,  188, 124, 201, 29, 199, 131, 226, 10 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -46);
  }

  // see row 11 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -34.70565203313315;
    const double sigma = 1.7009387219711465;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 69,  68, 60, 89,  87, 76, 44,  59, 7,   226, 225,
                          217, 7,  30, 109, 19, 61, 190, 50, 117, 75,  10 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -34);
  }

  // see row 12 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -44.36009577368896;
    const double sigma = 1.7009387219711465;
    const double sigmin = 1.277833697;

    uint8_t bytes[44] = {
      106, 193, 22,  237, 96,  194, 88,  226, 203, 174, 171, 114, 140, 72,  35,
      230, 218, 54,  225, 141, 8,   218, 93,  12,  193, 4,   226, 28,  199, 253,
      31,  92,  168, 217, 219, 182, 117, 38,  108, 146, 132, 72,  5,   158
    };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -44);
  }

  // see row 13 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -21.783037079346236;
    const double sigma = 1.6958406126012802;
    const double sigmin = 1.277833697;

    uint8_t bytes[11] = { 104, 22, 59, 193, 226, 203, 243, 225, 142, 116, 38 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -23);
  }

  // see row 14 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -39.68827784633828;
    const double sigma = 1.6958406126012802;
    const double sigmin = 1.277833697;

    uint8_t bytes[11] = { 214, 161, 181, 29, 118, 34, 42, 112, 90, 2, 89 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -40);
  }

  // see row 15 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -18.488607061056847;
    const double sigma = 1.6955259305261838;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 240, 82,  59, 250, 168, 163, 148, 191, 78,  165, 193,
                          15,  132, 35, 102, 253, 226, 134, 214, 163, 8,   3 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -22);
  }

  // see row 16 of table 3.2 in falcon specification
  // https://falcon-sign.info/falcon.pdf
  {
    const double mu = -48.39610939101591;
    const double sigma = 1.6955259305261838;
    const double sigmin = 1.277833697;

    uint8_t bytes[22] = { 135, 189, 135, 230, 51,  116, 206, 230, 33,  39, 252,
                          105, 49,  16,  74,  171, 100, 241, 54,  160, 72, 91 };

    int32_t z = samplerz::samplerz(mu, sigma, sigmin, bytes);
    assert(z == -50);
  }
}

}
