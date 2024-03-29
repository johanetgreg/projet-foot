# Generated by Django 5.0 on 2023-12-22 10:42

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Joueur',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nom', models.CharField(max_length=100)),
                ('id_joueur', models.IntegerField()),
                ('age', models.IntegerField()),
                ('taille', models.FloatField()),
                ('masse', models.FloatField()),
                ('poste', models.CharField(max_length=50)),
                ('matchs', models.IntegerField()),
                ('mtitu', models.IntegerField()),
                ('mins', models.IntegerField()),
                ('but', models.IntegerField()),
                ('peno', models.IntegerField()),
                ('assists', models.IntegerField()),
                ('passes', models.IntegerField()),
                ('passekey', models.IntegerField()),
                ('tacle', models.IntegerField()),
                ('block', models.IntegerField()),
                ('interceptions', models.IntegerField()),
                ('duels_gagnes', models.IntegerField()),
                ('duels', models.IntegerField()),
                ('dribbles_reussis', models.IntegerField()),
                ('dribbles', models.IntegerField()),
                ('faute_obtenue', models.IntegerField()),
                ('faute_commise', models.IntegerField()),
                ('cjaune', models.IntegerField()),
                ('crouge', models.IntegerField()),
            ],
        ),
    ]
